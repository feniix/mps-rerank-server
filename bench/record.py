#!/usr/bin/env python3
"""Recording proxy for real rerank requests. Stdlib only.

    python3 bench/record.py --out bench/requests/mine            # listens on 127.0.0.1:8082
    HINDSIGHT_API_RERANKER_TEI_URL=http://127.0.0.1:8082 ...     # point Hindsight at it

Forwards every call to the server (--upstream, default http://127.0.0.1:8081)
and saves each POST /rerank body as one JSON file, so the files hold exactly
what Hindsight sends: the query and the candidate texts in the order Hindsight
built them, up to its HINDSIGHT_API_RERANKER_MAX_CANDIDATES. Recall keeps
working while it records.

The files contain your memory text. They stay on your machine: bench/requests/
is git-ignored. Share a length profile instead (bench/profile.py).
"""
import argparse
import itertools
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser(description="Recording proxy for real rerank requests.")
ap.add_argument("--listen", default="127.0.0.1:8082", help="host:port to listen on")
ap.add_argument("--upstream", default="http://127.0.0.1:8081", help="the rerank server")
ap.add_argument("--out", required=True, help="directory for the request files")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
os.chmod(args.out, 0o700)
counter = itertools.count(len(os.listdir(args.out)) + 1)
lock = threading.Lock()


def forward(method, path, body=None, headers=None):
    req = urllib.request.Request(args.upstream + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(*forward("GET", self.path))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/rerank":
            with lock:
                n = next(counter)
            path = os.path.join(args.out, f"req-{n:05d}.json")
            with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
                f.write(body)
        self._reply(*forward("POST", self.path, body, {"Content-Type": "application/json"}))

    def log_message(self, format, *a):
        pass


host, port = args.listen.rsplit(":", 1)
print(f"recording to {args.out}; listening on http://{host}:{port} -> {args.upstream}", flush=True)
ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
