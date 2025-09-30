"""Google Cloud Storage utilities for Metamon checkpoints.

This module implements two complementary integration approaches for persisting
Metamon/AMAGO checkpoints in Google Cloud Storage (GCS):

``GCSCheckpointManager``
    Handles the low-level upload/download logic and implements retention for
    local checkpoints.  It mirrors the exact directory structure produced by
    AMAGO so that resuming training works without any additional bookkeeping.

``train_with_gcs_checkpoints``
    A convenience wrapper around ``experiment.learn()`` that monkey-patches the
    experiment's ``save_checkpoint`` method during the call to add GCS uploads
    and local cleanup.  This approach requires **no changes** to existing
    training scripts beyond swapping the ``learn()`` call for the wrapper.

``patch_experiment_for_gcs``
    A reusable monkey-patch helper that can be used when direct control over the
    training loop is preferred.  This function returns a callable that undoes the
    monkey patch and can be used together with a ``try/finally`` block.

The module purposefully avoids importing ``google`` packages at module import
time so that the rest of the Metamon package remains importable in
environments without the Google Cloud SDK.  All interactions with the bucket
use ``typing.Any`` in their type hints, but the runtime still expects the object
to behave like ``google.cloud.storage.bucket.Bucket``.

Example
-------

.. code-block:: python

    from google.colab import auth
    from google.cloud import storage

    from metamon.rl.gcs_checkpoint import (
        GCSCheckpointManager,
        train_with_gcs_checkpoints,
    )

    # 1) Authenticate and configure Google Cloud Storage
    auth.authenticate_user()
    storage_client = storage.Client(project="your-project-id")
    bucket = storage_client.bucket("metamon-checkpoints")

    # 2) Build the AMAGO experiment as usual
    experiment = create_offline_rl_trainer(
        ckpt_dir="./local_checkpoints",
        run_name="my_metamon_run",
        # ... other configuration ...
    )

    # 3) Set up the checkpoint manager
    gcs_manager = GCSCheckpointManager(
        bucket=bucket,
        run_name="my_metamon_run",
        gcs_base_path="training-runs",
        local_ckpt_dir="./local_checkpoints",
    )

    # 4) Train with automatic GCS uploads every 5 epochs, keeping the
    #    two most recent checkpoints on local disk.
    experiment.start()
    train_with_gcs_checkpoints(
        gcs_manager=gcs_manager,
        experiment=experiment,
        gcs_upload_every_n_epochs=5,
        keep_local_checkpoints=2,
    )

    # 5) Resume later by downloading a checkpoint from GCS
    gcs_manager.download_checkpoint(epoch=20)
    experiment.load_checkpoint(epoch=20, resume_training_state=True)
    train_with_gcs_checkpoints(gcs_manager=gcs_manager, experiment=experiment)
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

__all__ = [
    "GCSCheckpointManager",
    "UploadResult",
    "train_with_gcs_checkpoints",
    "patch_experiment_for_gcs",
]

LOGGER = logging.getLogger(__name__)


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class UploadResult:
    """Summary describing a completed checkpoint upload."""

    epoch: int
    uploaded_files: Sequence[str]
    local_training_state: Path
    local_policy_path: Optional[Path]


class GCSCheckpointManager:
    """Manage uploads, downloads, and retention of Metamon checkpoints on GCS.

    Parameters
    ----------
    bucket:
        A ``google.cloud.storage.bucket.Bucket`` instance (typed as ``Any`` to
        avoid importing the dependency at module scope).
    run_name:
        Run identifier used when constructing the AMAGO experiment.
    gcs_base_path:
        Root prefix inside the bucket.  The resulting object layout will be
        ``{gcs_base_path}/{run_name}/ckpts/...``.
    local_ckpt_dir:
        Optional default location of the AMAGO checkpoint directory.  Individual
        method calls may override this path.
    max_retries:
        Number of times network operations should be retried.
    retry_base_delay:
        Initial delay between retries.  The delay is doubled after every failed
        attempt.
    chunk_size:
        Optional chunk size (in bytes) for resumable uploads/downloads.  When
        ``None`` the default GCS client chunking behaviour is used.
    logger:
        Optional ``logging.Logger`` instance for emitting status updates.
    """

    def __init__(
        self,
        *,
        bucket: Any,
        run_name: str,
        gcs_base_path: str,
        local_ckpt_dir: Optional[str] = None,
        max_retries: int = 5,
        retry_base_delay: float = 5.0,
        chunk_size: Optional[int] = 1024 * 1024 * 256,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not run_name:
            raise ValueError("run_name must be provided")

        self.bucket = bucket
        self.run_name = run_name
        self.gcs_base_path = gcs_base_path.strip("/")
        self.local_ckpt_dir = local_ckpt_dir
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.chunk_size = chunk_size
        self.logger = logger or LOGGER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upload_checkpoint(
        self,
        epoch: int,
        local_ckpt_dir: Optional[str] = None,
    ) -> UploadResult:
        """Upload the checkpoint for ``epoch`` from ``local_ckpt_dir`` to GCS.

        Parameters
        ----------
        epoch:
            Epoch identifier of the checkpoint to upload.
        local_ckpt_dir:
            Optional override for the local checkpoint root.  Defaults to the
            path supplied when constructing the manager.
        """

        local_root = self._resolve_local_root(local_ckpt_dir)
        training_state = self._resolve_training_state_dir(local_root, epoch)
        policy_path = self._resolve_policy_path(local_root, epoch)

        upload_pairs = list(self._iter_local_files_to_upload(local_root, training_state))
        if policy_path is not None and policy_path.is_file():
            upload_pairs.append((policy_path, self._blob_for_path(local_root, policy_path)))
        elif policy_path is None:
            self.logger.warning(
                "Policy weights for epoch %s were not found locally. Only the training state will be uploaded.",
                epoch,
            )

        if not upload_pairs:
            raise FileNotFoundError(
                f"No files discovered for epoch {epoch} under {training_state}."
            )

        self.logger.info(
            "Uploading epoch %s checkpoint to gs://%s/%s",
            epoch,
            getattr(self.bucket, "name", "<unknown>"),
            self._experiment_prefix,
        )

        uploaded: List[str] = []
        for file_path, blob_name in tqdm(
            upload_pairs,
            desc=f"Uploading epoch {epoch} checkpoint",
            unit="file",
        ):
            self._upload_file(file_path, blob_name)
            uploaded.append(blob_name)

        self.logger.info(
            "Completed upload for epoch %s (%s files)",
            epoch,
            len(uploaded),
        )

        return UploadResult(
            epoch=epoch,
            uploaded_files=uploaded,
            local_training_state=training_state,
            local_policy_path=policy_path,
        )

    def cleanup_old_local_checkpoints(
        self,
        *,
        local_ckpt_dir: Optional[str] = None,
        keep_n: int,
    ) -> List[Path]:
        """Keep only the ``keep_n`` most recent checkpoints on local disk.

        Parameters
        ----------
        local_ckpt_dir:
            Optional override for the local checkpoint root.  When omitted the
            manager's default path is used.
        keep_n:
            Number of checkpoints to retain locally.  Negative values are
            treated as ``0``.
        """

        keep_n = max(0, int(keep_n))
        local_root = self._resolve_local_root(local_ckpt_dir)
        training_root = local_root / self.run_name / "ckpts" / "training_states"
        if not training_root.exists():
            return []

        entries: List[Tuple[int, Path]] = []
        for path in training_root.iterdir():
            if not path.is_dir():
                continue
            epoch = self._extract_epoch_from_name(path.name)
            if epoch is None:
                continue
            entries.append((epoch, path))

        entries.sort(key=lambda item: item[0])
        if keep_n >= len(entries):
            return []

        to_delete = entries[: len(entries) - keep_n]
        removed: List[Path] = []
        for epoch, directory in to_delete:
            self.logger.info("Pruning local checkpoint for epoch %s at %s", epoch, directory)
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=False)
            removed.append(directory)

            policy_path = self._resolve_policy_path(local_root, epoch)
            if policy_path is not None and policy_path.exists():
                try:
                    policy_path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    self.logger.debug("Removed policy weights for epoch %s at %s", epoch, policy_path)

        return removed

    def download_checkpoint(
        self,
        *,
        epoch: int,
        local_ckpt_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[Path]:
        """Download ``epoch`` checkpoint from GCS into ``local_ckpt_dir``.

        Parameters
        ----------
        epoch:
            Epoch to download.
        local_ckpt_dir:
            Optional override for the local checkpoint root directory.
        overwrite:
            Whether to replace existing local files.
        """

        local_root = self._resolve_local_root(local_ckpt_dir)
        prefix = self._training_state_prefix(epoch)
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        if not blobs:
            raise FileNotFoundError(
                f"No checkpoint blobs found for epoch {epoch} (prefix={prefix})."
            )

        downloaded: List[Path] = []
        prefix = self._gcs_prefix
        for blob in tqdm(blobs, desc=f"Downloading epoch {epoch} checkpoint", unit="file"):
            if prefix and blob.name.startswith(prefix):
                relative = blob.name[len(prefix) :]
            else:
                relative = blob.name
            destination = local_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and destination.exists():
                self.logger.debug("Skipping existing file %s", destination)
                continue
            self._download_blob(blob, destination)
            downloaded.append(destination)

        policy_blob_name = self._policy_blob_name(epoch)
        policy_blob = self.bucket.blob(policy_blob_name)
        if policy_blob.exists():
            policy_destination = local_root / self._relative_policy_path(epoch)
            policy_destination.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not policy_destination.exists():
                self._download_blob(policy_blob, policy_destination)
                downloaded.append(policy_destination)

        return downloaded

    def list_available_checkpoints(self) -> List[int]:
        """Return the epochs that have checkpoints stored remotely."""

        prefix = self._training_states_prefix
        epochs = set()
        for blob in self.bucket.list_blobs(prefix=prefix):
            suffix = blob.name[len(prefix) :]
            if not suffix:
                continue
            directory = suffix.split("/", maxsplit=1)[0]
            epoch = self._extract_epoch_from_name(directory)
            if epoch is not None:
                epochs.add(epoch)
        return sorted(epochs)

    def should_upload(self, epoch: int, every_n: int) -> bool:
        """Return ``True`` when ``epoch`` should trigger an upload."""

        if every_n <= 0:
            return False
        return epoch % every_n == 0

    # ------------------------------------------------------------------
    # Monkey patch helpers
    # ------------------------------------------------------------------
    def create_save_hook(
        self,
        *,
        experiment: Any,
        gcs_upload_every_n_epochs: int,
        keep_local_checkpoints: int,
    ) -> Callable[[], None]:
        """Monkey-patch ``experiment.save_checkpoint`` to upload to GCS.

        Returns
        -------
        Callable[[], None]
            A function that, when called, restores the original
            ``save_checkpoint`` method.
        """

        original_save_checkpoint = getattr(experiment, "save_checkpoint")
        ckpt_dir = getattr(experiment, "ckpt_dir", None)
        if ckpt_dir is None:
            raise AttributeError(
                "Experiment does not expose 'ckpt_dir'; cannot enable GCS checkpointing."
            )
        if self.local_ckpt_dir is None:
            self.local_ckpt_dir = ckpt_dir

        checkpoint_counter = 0

        def patched_save_checkpoint(*args: Any, **kwargs: Any) -> Any:
            nonlocal checkpoint_counter
            result = original_save_checkpoint(*args, **kwargs)
            checkpoint_counter += 1
            setattr(experiment, "gcs_checkpoint_counter", checkpoint_counter)

            epoch = getattr(experiment, "epoch", None)
            if epoch is None:
                return result
            if not self.should_upload(checkpoint_counter, gcs_upload_every_n_epochs):
                return result

            try:
                upload_result = self.upload_checkpoint(epoch, ckpt_dir)
            except Exception as err:  # noqa: BLE001 - surfaced via logging after retries
                self.logger.exception(
                    "Failed to upload checkpoint for epoch %s: %s",
                    epoch,
                    err,
                )
                return result

            if keep_local_checkpoints >= 0:
                try:
                    self.cleanup_old_local_checkpoints(
                        local_ckpt_dir=ckpt_dir,
                        keep_n=max(keep_local_checkpoints, 0),
                    )
                except Exception:  # noqa: BLE001
                    self.logger.exception(
                        "Failed to prune local checkpoints after uploading epoch %s",
                        epoch,
                    )

            self.logger.debug(
                "Uploaded %s blobs for epoch %s", len(upload_result.uploaded_files), epoch
            )
            return result

        setattr(experiment, "save_checkpoint", patched_save_checkpoint)

        def restore() -> None:
            setattr(experiment, "save_checkpoint", original_save_checkpoint)

        return restore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _gcs_prefix(self) -> str:
        return f"{self.gcs_base_path}/" if self.gcs_base_path else ""

    @property
    def _experiment_prefix(self) -> str:
        return f"{self._gcs_prefix}{self.run_name}/ckpts/"

    @property
    def _training_states_prefix(self) -> str:
        return f"{self._experiment_prefix}training_states/"

    def _training_state_prefix(self, epoch: int) -> str:
        return f"{self._training_states_prefix}{self._training_state_dirname(epoch)}/"

    def _training_state_dirname(self, epoch: int) -> str:
        return f"{self.run_name}_epoch_{epoch}"

    def _training_state_dir(self, local_root: Path) -> Path:
        return local_root / self.run_name / "ckpts" / "training_states"

    def _resolve_training_state_dir(self, local_root: Path, epoch: int) -> Path:
        directory = self._training_state_dir(local_root) / self._training_state_dirname(epoch)
        if not directory.exists():
            raise FileNotFoundError(
                f"Training state for epoch {epoch} not found at {directory}."
            )
        return directory

    def _resolve_policy_path(self, local_root: Path, epoch: int) -> Optional[Path]:
        policy_dir = local_root / self.run_name / "ckpts" / "policy_weights"
        path = policy_dir / f"policy_epoch_{epoch}.pt"
        return path if path.exists() else None

    def _relative_policy_path(self, epoch: int) -> Path:
        return Path(self.run_name) / "ckpts" / "policy_weights" / f"policy_epoch_{epoch}.pt"

    def _policy_blob_name(self, epoch: int) -> str:
        return f"{self._gcs_prefix}{self._relative_policy_path(epoch).as_posix()}"

    def _blob_for_path(self, local_root: Path, file_path: Path) -> str:
        relative = file_path.resolve().relative_to(local_root)
        return f"{self._gcs_prefix}{relative.as_posix()}"

    def _resolve_local_root(self, override: Optional[str]) -> Path:
        path = override or self.local_ckpt_dir
        if path is None:
            raise ValueError(
                "local_ckpt_dir must be provided either at initialisation or per method call."
            )
        return _expand(path)

    def _iter_local_files_to_upload(
        self,
        local_root: Path,
        directory: Path,
    ) -> Iterator[Tuple[Path, str]]:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                yield file_path, self._blob_for_path(local_root, file_path)

    def _upload_file(self, file_path: Path, blob_name: str) -> None:
        def _operation() -> None:
            blob = self.bucket.blob(blob_name)
            if self.chunk_size is not None:
                blob.chunk_size = self.chunk_size
            blob.upload_from_filename(file_path.as_posix())

        self._retry_with_backoff(
            operation=_operation,
            description=f"upload {file_path} -> {blob_name}",
        )

    def _download_blob(self, blob: Any, destination: Path) -> None:
        def _operation() -> None:
            if self.chunk_size is not None:
                blob.chunk_size = self.chunk_size
            blob.download_to_filename(destination.as_posix())

        self._retry_with_backoff(
            operation=_operation,
            description=f"download {blob.name} -> {destination}",
        )

    def _retry_with_backoff(
        self,
        *,
        operation: Callable[[], None],
        description: str,
    ) -> None:
        delay = self.retry_base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                operation()
                return
            except Exception as err:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to {description} after {self.max_retries} attempts"
                    ) from err
                self.logger.warning(
                    "Attempt %s/%s to %s failed (%s). Retrying in %.1f seconds...",
                    attempt,
                    self.max_retries,
                    description,
                    err,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def _extract_epoch_from_name(name: str) -> Optional[int]:
        parts = name.split("_")
        for index, part in enumerate(parts):
            if part == "epoch" and index + 1 < len(parts):
                try:
                    return int(parts[index + 1])
                except ValueError:
                    return None
        return None


def train_with_gcs_checkpoints(
    *,
    gcs_manager: GCSCheckpointManager,
    experiment: Any,
    gcs_upload_every_n_epochs: int = 5,
    keep_local_checkpoints: int = 2,
) -> None:
    """Run ``experiment.learn()`` while synchronising checkpoints to GCS.

    Parameters
    ----------
    gcs_manager:
        Instance of :class:`GCSCheckpointManager` configured for the run.
    experiment:
        AMAGO experiment returned by ``create_offline_rl_trainer``.
    gcs_upload_every_n_epochs:
        Upload cadence.  Set to ``0`` or a negative value to disable remote
        uploads while still using the wrapper.
    keep_local_checkpoints:
        Number of most recent checkpoints to keep locally.  A negative value
        disables cleanup entirely.
    """

    restore = gcs_manager.create_save_hook(
        experiment=experiment,
        gcs_upload_every_n_epochs=gcs_upload_every_n_epochs,
        keep_local_checkpoints=keep_local_checkpoints,
    )
    try:
        experiment.learn()
    finally:
        restore()


def patch_experiment_for_gcs(
    *,
    gcs_manager: GCSCheckpointManager,
    experiment: Any,
    gcs_upload_every_n_epochs: int = 5,
    keep_local_checkpoints: int = 2,
) -> Callable[[], None]:
    """Patch ``experiment.save_checkpoint`` to add GCS synchronisation.

    This helper mirrors the behaviour of :func:`train_with_gcs_checkpoints` but
    only performs the monkey patch.  The returned callable restores the original
    method, allowing callers to control when the patch is active.
    """

    return gcs_manager.create_save_hook(
        experiment=experiment,
        gcs_upload_every_n_epochs=gcs_upload_every_n_epochs,
        keep_local_checkpoints=keep_local_checkpoints,
    )


# Backwards compatibility aliases for readability in notebooks.
GCSCheckpointPatch = patch_experiment_for_gcs

