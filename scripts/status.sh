#!/bin/sh
# Show the launchd agent's state and what the server reports at /info.
label="io.github.feniix.mps-rerank-server"
port="${RERANKER_PORT:-8081}"
launchctl print "gui/$(id -u)/$label" 2>/dev/null | grep -E "state =|pid =|last exit" || echo "agent not loaded"
curl -sf "http://127.0.0.1:$port/info" && echo || echo "server not responding on :$port (first start can take minutes)"
