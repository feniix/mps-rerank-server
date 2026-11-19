#!/bin/sh
# Install (or restart) mps-rerank-server as a per-user launchd agent.
#
#   scripts/install.sh                      # 127.0.0.1:8081
#   RERANKER_PORT=8090 scripts/install.sh   # another port
#
# Renders launchd/io.github.feniix.mps-rerank-server.plist into
# ~/Library/LaunchAgents, lints it, stops any running copy and loads the new
# one. Re-run it after moving this directory, after upgrading uv (the plist
# holds uv's absolute path), or to restart the server.
set -eu

label="io.github.feniix.mps-rerank-server"
dir="$(cd "$(dirname "$0")/.." && pwd)"
plist="$HOME/Library/LaunchAgents/$label.plist"
domain="gui/$(id -u)"
host="${RERANKER_HOST:-127.0.0.1}"
port="${RERANKER_PORT:-8081}"

uv="$(command -v uv || true)"
if [ -z "$uv" ]; then
  echo "uv not found on PATH; install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "warning: MPS needs an Apple Silicon Mac; the server will fall back to CPU" >&2
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__UV__|$uv|g" -e "s|__DIR__|$dir|g" -e "s|__HOME__|$HOME|g" \
    -e "s|__HOST__|$host|g" -e "s|__PORT__|$port|g" \
    "$dir/launchd/$label.plist" > "$plist"
plutil -lint "$plist" >/dev/null

launchctl bootout "$domain/$label" 2>/dev/null || true
# bootout returns before the agent is gone; bootstrapping too soon fails with
# "Bootstrap failed: 5: Input/output error" and leaves the server down.
i=0
while launchctl print "$domain/$label" >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 30 ] && { echo "agent did not stop within 30 s" >&2; exit 1; }
  sleep 1
done
launchctl bootstrap "$domain" "$plist"

echo "Installed $label on http://$host:$port"
echo "Logs: ~/Library/Logs/mps-rerank-server.log"
echo "The first start downloads PyTorch and the model; check with scripts/status.sh."
