#!/usr/bin/env bash
set -euo pipefail

# --- Config (edit if you prefer HTTPS for the main clone) ---
METAMON_SSH_REPO="git@github.com:nivaldoh/metamon.git"
AMAGO_HTTPS_REPO="https://github.com/UT-Austin-RPL/amago.git"

# --- Clone Metamon (with submodules) ---
if [ ! -d "metamon/.git" ]; then
  git clone --recurse-submodules "${METAMON_SSH_REPO}" metamon
else
  echo "[info] metamon already exists; updating"
  (cd metamon && git fetch --all && git pull --rebase)
fi

# Ensure submodule(s) are initialized & up-to-date (handles re-runs)
(
  cd metamon
  git submodule sync --recursive
  git submodule update --init --recursive
)

# --- Install Metamon (editable) ---
(
  cd metamon
  python -m pip install -U pip
  python -m pip install -e .
)

# --- Clone AMAGO into Metamon root and install (editable) ---
if [ ! -d "metamon/amago/.git" ]; then
  git clone "${AMAGO_HTTPS_REPO}" metamon/amago
else
  echo "[info] amago already exists under metamon/; updating"
  (cd metamon/amago && git fetch --all && git pull --rebase)
fi

(
  cd metamon
  python -m pip install -e ./amago
)

# --- Set up Pokemon Showdown server (submodule) ---
(
  cd metamon/server/pokemon-showdown
  # Prefer reproducible installs if package-lock.json exists
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi

  # Start server in background (default port: 8000)
  # --no-security is typical for local dev; remove for stricter mode
  nohup node pokemon-showdown start --no-security > ../../ps.log 2>&1 &
  echo "[info] Pokemon Showdown started (logs: metamon/ps.log)"
)

# --- Optional: use local cache or GCS fuse for datasets ---
export METAMON_CACHE_DIR=/cache
echo "[info] METAMON_CACHE_DIR=${METAMON_CACHE_DIR}"
echo "[done] Metamon + AMAGO set up. Pokemon Showdown is running."
