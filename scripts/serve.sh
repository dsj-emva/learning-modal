#!/bin/sh
# Start the Streamlit app on $PORT (default 8080). Container entrypoint (Dockerfile CMD, Railway start command).
# Railway mounts volumes owned by root. If the container was started as root (Railway: RAILWAY_RUN_UID=0),
# hand DATA_DIR to the unprivileged app user (uid 1000) and drop to it; otherwise refuse to start when
# DATA_DIR is not writable, instead of failing later on the first upload.
set -eu
DATA_DIR="${DATA_DIR:-/data}"
PORT="${PORT:-8080}"
set -- python -m streamlit run app/main.py --server.port "$PORT" --server.address 0.0.0.0
mkdir -p "$DATA_DIR"
if [ "$(id -u)" = "0" ]; then
    chown -R 1000:1000 "$DATA_DIR"
    exec setpriv --reuid=1000 --regid=1000 --init-groups env HOME=/home/app "$@"
fi
if [ ! -w "$DATA_DIR" ]; then
    echo "serve.sh: DATA_DIR=$DATA_DIR is not writable by uid $(id -u)." >&2
    echo "On Railway the volume is root-owned: set RAILWAY_RUN_UID=0 (this script then drops to uid 1000)." >&2
    exit 1
fi
exec "$@"
