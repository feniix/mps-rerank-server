# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "torch==2.14.0",
#     "sentence-transformers==6.1.0",
#     "transformers==5.17.0",
# ]
# ///
"""TEI-compatible cross-encoder reranker on Apple Silicon GPUs (PyTorch MPS).

Hindsight's built-in `local` reranker runs on the CPU on a Mac: inside Docker
the Linux VM has no Metal access, and the native install never selects MPS.
This serves a cross-encoder natively on the GPU and implements the two calls
Hindsight's `tei` reranker provider makes:

  GET  /info    -> {"model_id", "revision", "device", "dtype"}
  POST /rerank  {"query", "texts"} -> [{"index", "score"}], best first

GET /health returns {"status": "ok"}. Scores are raw logits, as the `local`
provider returns them; Hindsight applies the sigmoid itself.

Default model: hindsight-ai/hs-rerank-1 in FP16, pinned to a revision. It is
Hindsight's own default reranker and reads up to 8,192 tokens per pair.

Run with `uv run --locked --script server.py`; versions are pinned above and
locked in server.py.lock. `scripts/install.sh` installs it as a launchd
agent (see README.md).
"""

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from sentence_transformers import CrossEncoder

# Pinned to a revision so an upstream re-upload cannot change rankings underneath
# you. Both can be overridden to serve another cross-encoder.
MODEL_ID = os.environ.get("RERANKER_MODEL_ID", "hindsight-ai/hs-rerank-1")
MODEL_REVISION = os.environ.get("RERANKER_MODEL_REVISION", "927f5be31280298a51cf29af8a29cb111ed812c5")
# float16 for hs-rerank-1 (its weights ship in FP16); float32 for MiniLM-L6.
DTYPE = os.environ.get("RERANKER_DTYPE", "float16")
# Loopback only by default: no authentication, so do not expose it on the LAN.
# OrbStack routes host.docker.internal to the host's 127.0.0.1, so a Hindsight
# container reaches it without binding to other interfaces.
HOST = os.environ.get("RERANKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("RERANKER_PORT", "8081"))
BATCH_SIZE = int(os.environ.get("RERANKER_BATCH_SIZE", "32"))
# MPS keeps the buffers it allocates for each input shape, and hs-rerank-1's
# long, variable-length inputs grow that pool without bound (7.5 GB footprint
# after ~46 h). Hand it back to the driver on this cadence; 0 disables. It does
# not clear MPS's per-shape graph cache (no API for that), so the footprint
# still creeps; restart the agent (`scripts/install.sh`) to reset it.
EMPTY_CACHE_SECS = float(os.environ.get("RERANKER_EMPTY_CACHE_SECS", "3600"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mps-rerank-server")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if DEVICE == "cpu":
    log.warning("MPS unavailable; serving on CPU")

# Identity, not the model's own activation: hs-rerank-1 ships a Sigmoid, and
# Hindsight sigmoids TEI scores itself, so a second one would squash them.
model = CrossEncoder(
    MODEL_ID,
    revision=MODEL_REVISION,
    device=DEVICE,
    activation_fn=torch.nn.Identity(),
    model_kwargs={"torch_dtype": getattr(torch, DTYPE)},
)
# One forward pass at a time on the GPU; concurrent requests queue here.
predict_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/info":
            self._send(200, {"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": DEVICE, "dtype": DTYPE})
        elif self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            query, texts = req["query"], req["texts"]
            if not isinstance(query, str) or not isinstance(texts, list):
                raise TypeError("query must be a string and texts a list")
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        if not texts:
            self._send(200, [])
            return
        with predict_lock:
            # No progress bar: at INFO it would write one to the launchd log per recall.
            scores = model.predict(
                [(query, t) for t in texts], batch_size=BATCH_SIZE, show_progress_bar=False
            )
        ranked = sorted(enumerate(scores.tolist()), key=lambda pair: -pair[1])
        self._send(200, [{"index": i, "score": s} for i, s in ranked])

    def log_message(self, format: str, *args: object) -> None:
        # Per-request access logs would flood the launchd log on every recall.
        pass


def empty_cache_periodically() -> None:
    while True:
        time.sleep(EMPTY_CACHE_SECS)
        # Under the lock, so the cache is never pulled out from under a forward pass.
        with predict_lock:
            before = torch.mps.driver_allocated_memory()
            torch.mps.synchronize()
            torch.mps.empty_cache()
            after = torch.mps.driver_allocated_memory()
        log.info("emptied MPS cache: driver %d -> %d MiB", before >> 20, after >> 20)


def main() -> None:
    # Warm up so the first recall does not pay for MPS kernel compilation.
    model.predict([("warm up", "warm up")], show_progress_bar=False)
    if DEVICE == "mps" and EMPTY_CACHE_SECS > 0:
        threading.Thread(target=empty_cache_periodically, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    log.info("serving %s@%s (%s) on %s at http://%s:%d", MODEL_ID, MODEL_REVISION[:8], DTYPE, DEVICE, HOST, PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
