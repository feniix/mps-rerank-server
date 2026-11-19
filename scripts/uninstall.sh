#!/bin/sh
# Stop mps-rerank-server and remove its launchd agent. The uv cache and the
# Hugging Face model cache are left in place.
set -eu
label="io.github.feniix.mps-rerank-server"
domain="gui/$(id -u)"
launchctl bootout "$domain/$label" 2>/dev/null || true
# bootout returns before launchd has removed the agent; wait so a following
# install or status sees the real state.
i=0
while launchctl print "$domain/$label" >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 30 ] && { echo "agent did not stop within 30 s" >&2; exit 1; }
  sleep 1
done
rm -f "$HOME/Library/LaunchAgents/$label.plist"
echo "Uninstalled $label"
