#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check that the vocabulary file exists
if [ ! -f "static_vocabulary.txt" ]; then
    echo "ERROR: static_vocabulary.txt not found in $SCRIPT_DIR"
    echo "       This file is required for the chat server."
    exit 1
fi

# Check if the binary exists; if not, build it
BINARY="target/release/chat_server"
if [ ! -f "$BINARY" ]; then
    echo "  Building chat server (this may take a few minutes) ..."
    cargo build --release 2>&1
    echo ""
fi

echo "  Starting WormBrain Chat Server ..."
echo "  Open http://localhost:${1:-8080} in your browser."
echo "  Press Ctrl+C to stop."
echo ""

exec "$BINARY" "${@}"
