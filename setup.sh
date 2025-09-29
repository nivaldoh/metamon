#!/usr/bin/env bash
set -euo pipefail

# --- Config (edit if you prefer HTTPS for the main clone) ---
METAMON_SSH_REPO="git@github.com:nivaldoh/metamon.git"
AMAGO_HTTPS_REPO="https://github.com/UT-Austin-RPL/amago.git"

# --- Clone Metamon (with submodules) ---
# Do this before running the script
# git clone --recurse-submodules "${METAMON_SSH_REPO}" metamon

# --- Install Metamon (editable) ---
(
  python -m pip install -U pip
  python -m pip install -e .
)

# --- Clone AMAGO into Metamon root and install (editable) ---
if [ ! -d "amago/.git" ]; then
  git clone "${AMAGO_HTTPS_REPO}" amago
else
  echo "[info] amago already exists under metamon/; updating"
  (cd amago && git fetch --all && git pull --rebase)
fi

(
  python -m pip install -e ./amago
  python -m pip install amago[flash]
)

# --- Set up Pokemon Showdown server (submodule) ---
(
  cd server/pokemon-showdown
  # Prefer reproducible installs if package-lock.json exists
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi

  # TODO: this gets stuck. Need to autoskip dialog
  # Start server in background (default port: 8000)
  #   nohup node pokemon-showdown start --no-security > ../../ps.log 2>&1 &
  #   echo "[info] Pokemon Showdown started (logs: metamon/ps.log)"
)

# --- Optional: use local cache or GCS fuse for datasets ---
export METAMON_CACHE_DIR=/cache
echo "[info] METAMON_CACHE_DIR=${METAMON_CACHE_DIR}"
echo "[done] Metamon + AMAGO set up. Pokemon Showdown is running."
