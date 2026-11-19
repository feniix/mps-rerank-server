#!/bin/sh
# Stop mps-rerank-server and remove its launchd agent. The uv cache and the
# Hugging Face model cache are left in place.
set -eu
label="io.github.feniix.mps-rerank-server"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$label.plist"
echo "Uninstalled $label"
