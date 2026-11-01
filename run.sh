#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -f "$DIR/.venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found at .venv"
    echo "Please set up the environment first:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    exit 1
fi
exec "$DIR/.venv/bin/python" "$DIR/asl_main.py" "$@"
