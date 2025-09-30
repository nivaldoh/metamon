import os
from functools import partial
from typing import List, Optional, Tuple

import wandb

import amago

import metamon
import metamon.rl
import torch
try:
    from accelerate.state import AcceleratorState
    from accelerate.utils import DistributedType
except Exception:  # pragma: no cover - accelerate is optional at runtime
    AcceleratorState = None  # type: ignore[assignment]
    DistributedType = None  # type: ignore[assignment]
from metamon.env import get_metamon_teams
from metamon.interface import (
    TokenizedObservationSpace,
    ActionSpace,
    RewardFunction,
)
from metamon.tokenizer import get_tokenizer
from metamon.data import ParsedReplayDataset
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGODataset,
    make_baseline_env,
    make_placeholder_env,
)
from metamon import baselines


WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")
EVAL_OPPONENTS = [
    baselines.heuristic.basic.PokeEnvHeuristic,
    baselines.heuristic.basic.Gen1BossAI,
    baselines.heuristic.basic.Grunt,
    baselines.heuristic.basic.GymLeader,
    baselines.heuristic.kaizo.EmeraldKaizo,
]


def _bf16_supported() -> bool:
    """Return True if the current device supports bf16 training."""

    if not torch.cuda.is_available():
        return False
    is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    try:
        return bool(is_supported())
    except Exception:
        return False


def _resolve_mixed_precision(requested: str) -> str:
    """Resolve the requested mixed precision mode based on hardware support."""

    requested = requested.lower()
    if requested not in {"no", "fp16", "bf16", "auto"}:
        raise ValueError(f"Unsupported mixed precision option: {requested}")

    if requested == "auto":
        if torch.cuda.is_available():
            if _bf16_supported():
                return "bf16"
            return "fp16"
        return "no"

    if requested == "bf16" and not _bf16_supported():
        print("[INFO] bf16 requested but not supported; falling back to fp16/no.")
        if torch.cuda.is_available():
            return "fp16"
        return "no"

    if requested == "fp16" and not torch.cuda.is_available():
        print("[INFO] fp16 requested but CUDA is unavailable; using full precision.")
        return "no"

    return requested


def _get_accelerator_state() -> Optional["AcceleratorState"]:
    """Return the current Accelerate state if initialized.

    The training stack internally constructs an :class:`accelerate.Accelerator` when
    needed, so we only query the state here to detect distributed launches without
    creating new accelerator instances.
    """

    if AcceleratorState is None:  # accelerate is not installed
        return None

    try:
        state = AcceleratorState()
    except Exception:  # pragma: no cover - defensive, mirrors accelerate internals
        return None

    # ``AcceleratorState`` defaults to a single-process configuration when the
    # script is executed without ``accelerate launch``. We return the state in all
    # cases so callers can still rely on helper properties like ``is_main_process``.
    return state


def add_cli(parser):
    parser.add_argument(
        "--run_name",
        required=True,
        help="Give the run a name to identify logs and checkpoints.",
    )
    parser.add_argument(
        "--obs_space",
        type=str,
        default="TeamPreviewObservationSpace",
        help="See the README for a description of the different observation spaces.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default="DefaultShapedReward",
        help="See the README for a description of the different reward functions.",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="DefaultActionSpace",
        help="See the README for a description of the different action spaces.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save checkpoints. Find checkpoints under {save_dir}/{run_name}/ckpts/",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Resume training from an existing run with this run_name. Provide the epoch checkpoint to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train for. In offline RL model, an epoch is an arbitrary interval (here: 25k) of training steps on a fixed dataset.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=12,
        help="Batch size per GPU. Total batch size is batch_size_per_gpu * num_gpus.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of gradient accumulations per update.",
    )
    parser.add_argument(
        "--model_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the model architecture. See provided rl/configs/models/",
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the training or hparams. See provided rl/configs/training/",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DefaultObservationSpace-v1",
        help="The tokenizer to use for the text observation space. See metamon.tokenizer for options.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=10,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--parsed_replay_dir",
        type=str,
        default=None,
        help="Path to the parsed replay directory. Defaults to the official huggingface version.",
    )
    parser.add_argument(
        "--custom_replay_dir",
        type=str,
        default=None,
        help="Path to an optional second parsed replay dataset (e.g., self-play data you've collected).",
    )
    parser.add_argument(
        "--custom_replay_sample_weight",
        type=float,
        default=0.25,
        help="[0, 1] portion of each batch to sample from the custom dataset (if provided).",
    )
    parser.add_argument(
        "--async_env_mp_context",
        type=str,
        default="forkserver",
        help="Async environment setup method. Options: 'forkserver' (recommended, fast), 'fork' (fastest but unsafe with threads), 'spawn' (slowest but safest). Use 'spawn' only if others hang.",
    )
    parser.add_argument(
        "--eval_gens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 9],
        help="Generations (of OU) to play against heuristics between training epochs. Win rates usually saturate at 90\%+ quickly, so this is mostly a sanity-check. Reduce gens to save time on launch! Use `--eval_gens` (no arguments) to disable evaluation.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Showdown battle formats to include in the dataset. Defaults to all supported formats.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16", "auto"],
        default="auto",
        help="Mixed precision mode to use. Defaults to automatic selection based on hardware.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimizations.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable automatic performance optimizations (mixed precision and torch.compile).",
    )
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
    return parser


def create_offline_dataset(
    obs_space: TokenizedObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    parsed_replay_dir: str,
    custom_replay_dir: Optional[str] = None,
    custom_replay_sample_weight: float = 0.25,
    verbose: bool = True,
    formats: Optional[List[str]] = None,
) -> amago.loading.RLDataset:

    formats = formats or metamon.SUPPORTED_BATTLE_FORMATS
    dset_kwargs = {
        "observation_space": obs_space,
        "action_space": action_space,
        "reward_function": reward_function,
        # amago will handle sequence lengths on its side
        "max_seq_len": None,
        "formats": formats,
        "verbose": verbose,  # False to hide dset setup progress bar
    }
    parsed_replays_amago = MetamonAMAGODataset(
        dset_name="Metamon Parsed Replays",
        parsed_replay_dset=ParsedReplayDataset(
            dset_root=parsed_replay_dir, **dset_kwargs
        ),
    )
    if custom_replay_dir is not None:
        custom_dset_amago = MetamonAMAGODataset(
            dset_name="Custom Parsed Replays",
            parsed_replay_dset=ParsedReplayDataset(
                dset_root=custom_replay_dir, **dset_kwargs
            ),
        )
        amago_dataset = amago.loading.MixtureOfDatasets(
            datasets=[parsed_replays_amago, custom_dset_amago],
            sampling_weights=[
                1 - custom_replay_sample_weight,
                custom_replay_sample_weight,
            ],
        )
    else:
        amago_dataset = parsed_replays_amago
    return amago_dataset


def create_offline_rl_trainer(
    ckpt_dir: str,
    run_name: str,
    model_gin_config: str,
    train_gin_config: str,
    obs_space: TokenizedObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    amago_dataset: amago.loading.Dataset,
    eval_gens: List[int],
    async_env_mp_context: str = "spawn",
    val_timesteps_per_epoch: int = 300,
    dloader_workers: int = 8,
    epochs: int = 40,
    grad_accum: int = 1,
    steps_per_epoch: int = 25_000,
    batch_size_per_gpu: int = 16,
    log: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    manual_gin_overrides: Optional[dict] = None,
    mixed_precision: str = "auto",
    enable_compile: bool = True,
) -> Tuple[MetamonAMAGOExperiment, Optional["AcceleratorState"]]:
    """
    Convenience function that creates an AMAGO experiment with default arguments
    set for offline RL in metamon. Returns the initialized experiment along with
    the active :class:`accelerate.state.AcceleratorState`, if any.
    """
    accelerator_state = _get_accelerator_state()
    num_processes = accelerator_state.num_processes if accelerator_state else 1
    is_distributed = (
        accelerator_state is not None
        and accelerator_state.distributed_type is not None
        and accelerator_state.distributed_type != getattr(DistributedType, "NO", None)
        and num_processes > 1
    )
    is_main_process = accelerator_state.is_main_process if accelerator_state else True

    if is_main_process:
        launch_mode = "Accelerate" if is_distributed else "single process"
        print(
            f"[INFO] Launch mode: {launch_mode} (num_processes={num_processes})"
        )
        print(f"[INFO] Per-process batch size: {batch_size_per_gpu}")
        print(
            f"[INFO] Effective global batch size: {batch_size_per_gpu * num_processes}"
        )

    # configuration
    config = {
        "MetamonTstepEncoder.tokenizer": obs_space.tokenizer,
        "MetamonPerceiverTstepEncoder.tokenizer": obs_space.tokenizer,
    }
    if manual_gin_overrides is not None:
        config.update(manual_gin_overrides)
    model_config_path = os.path.join(metamon.rl.MODEL_CONFIG_DIR, model_gin_config)
    training_config_path = os.path.join(
        metamon.rl.TRAINING_CONFIG_DIR, train_gin_config
    )
    amago.cli_utils.use_config(config, [model_config_path, training_config_path])

    # validation environments (evaluated throughout training)
    if eval_gens:
        print(f"[DEBUG] Creating validation environments for generations: {eval_gens}")
        print(f"[DEBUG] Number of opponent types: {len(EVAL_OPPONENTS)}")
        make_envs = [
            partial(
                make_baseline_env,
                battle_format=f"gen{gen}ou",
                observation_space=obs_space,
                action_space=action_space,
                reward_function=reward_function,
                team_set=get_metamon_teams(f"gen{gen}ou", "competitive"),
                opponent_type=opponent,
            )
            for gen in set(eval_gens)
            for opponent in EVAL_OPPONENTS
        ]
        print(f"[DEBUG] Total validation environments to create: {len(make_envs)}")
    else:
        print("[DEBUG] No eval_gens specified, creating placeholder environment")
        # turn off eval envs during training (do evals separately).
        make_envs = [partial(make_placeholder_env, obs_space, action_space)]
        val_timesteps_per_epoch = 0

    print("[DEBUG] Creating MetamonAMAGOExperiment...")
    # With lazy initialization, we can now use async mode for validation environments
    print(f"[DEBUG] Using env_mode=async for validation environments (lazy initialization enabled)")

    resolved_mixed_precision = _resolve_mixed_precision(mixed_precision)
    should_compile = enable_compile and hasattr(torch, "compile")

    if resolved_mixed_precision != "no":
        print(f"✓ Mixed precision enabled: {resolved_mixed_precision}")
    else:
        print("[INFO] Mixed precision disabled.")

    if enable_compile and not hasattr(torch, "compile"):
        print("[INFO] torch.compile unavailable in this PyTorch build; skipping compilation.")
        should_compile = False
    elif should_compile:
        print("✓ torch.compile optimizations enabled (mode=default)")
    else:
        print("[INFO] torch.compile disabled.")

    experiment = MetamonAMAGOExperiment(
        ## required ##
        run_name=run_name,
        ckpt_base_dir=ckpt_dir,
        # max_seq_len = should be set in the gin file
        dataset=amago_dataset,
        # tstep_encoder_type = should be set in the gin file
        # traj_encoder_type = should be set in the gin file
        # agent_type = should be set in the gin file
        val_timesteps_per_epoch=val_timesteps_per_epoch,  # per actor
        ## environment ##
        make_train_env=partial(make_placeholder_env, obs_space, action_space),
        make_val_env=make_envs,
        env_mode="async",  # Now we can use async mode thanks to lazy initialization
        async_env_mp_context=async_env_mp_context,
        parallel_actors=len(make_envs),
        # no exploration
        exploration_wrapper_type=None,
        sample_actions=True,
        force_reset_train_envs_every=None,
        ## logging ##
        log_to_wandb=log and is_main_process,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        verbose=True,
        log_interval=300,
        ## replay ##
        padded_sampling="none",
        dloader_workers=dloader_workers,
        ## learning schedule ##
        epochs=epochs,
        # entirely offline RL
        start_learning_at_epoch=0,
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        train_batches_per_epoch=steps_per_epoch * grad_accum,
        val_interval=1,
        ckpt_interval=2,
        ## optimization ##
        batch_size=batch_size_per_gpu,
        batches_per_update=grad_accum,
        mixed_precision=resolved_mixed_precision,
        compile_model=should_compile,
    )
    return experiment, accelerator_state


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.interface import (
        get_observation_space,
        get_reward_function,
        get_action_space,
    )

    parser = ArgumentParser(
        description="Train a Metamon RL agent from scratch using offline RL on parsed replay data. "
        "This script trains new models using imitation learning or reinforcement learning objectives "
        "on the dataset of human Pokémon battles (& an optional custom dataset of self-play data you've collected)."
    )
    add_cli(parser)
    args = parser.parse_args()

    # agent input/output/rewards
    obs_space = TokenizedObservationSpace(
        get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
    )
    reward_function = get_reward_function(args.reward_function)
    action_space = get_action_space(args.action_space)

    # metamon dataset
    amago_dataset = create_offline_dataset(
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        parsed_replay_dir=args.parsed_replay_dir,
        custom_replay_dir=args.custom_replay_dir,
        custom_replay_sample_weight=args.custom_replay_sample_weight,
        formats=args.formats,
    )

    # quick-setup for an offline RL experiment
    print(f"[DEBUG] Creating offline RL trainer with eval_gens={args.eval_gens}")
    experiment, accelerator_state = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=args.model_gin_config,
        train_gin_config=args.train_gin_config,
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        batch_size_per_gpu=args.batch_size_per_gpu,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        mixed_precision=("no" if args.no_optimize else args.mixed_precision),
        enable_compile=(
            (not args.no_optimize) and (not args.no_compile)
        ),
    )
    print("[DEBUG] Offline RL trainer created successfully")
    print("[DEBUG] Starting experiment initialization...")
    try:
        experiment.start()
        print("[DEBUG] Experiment started successfully")
    except Exception as e:
        print(f"[DEBUG ERROR] Failed to start experiment: {e}")
        import traceback
        traceback.print_exc()
        raise

    if args.ckpt is not None:
        # resume training from a checkpoint
        print(f"[DEBUG] Loading checkpoint: {args.ckpt}")
        experiment.load_checkpoint(args.ckpt)
        print("[DEBUG] Checkpoint loaded successfully")

    print("[DEBUG] Starting learning phase...")
    try:
        experiment.learn()
        print("[DEBUG] Learning completed successfully")
    except Exception as e:
        print(f"[DEBUG ERROR] Failed during learning: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Only finalize wandb on the process that initialized it.
    if wandb.run is not None and (
        accelerator_state is None or accelerator_state.is_main_process
    ):
        wandb.finish()
