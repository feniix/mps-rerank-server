"""API contract tests for server.py. Run: python3 -m unittest discover -s tests -v

Starts server.py with uv on a free port, serving the small MiniLM-L6
cross-encoder in FP32 so the test is quick; the API does not depend on the
model. Needs uv on PATH and network access the first time (PyTorch and the
model are downloaded and cached).
"""
import json
import os
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.url = f"http://127.0.0.1:{cls.port}"
        env = dict(os.environ, RERANKER_PORT=str(cls.port), RERANKER_DTYPE="float32",
                   RERANKER_MODEL_ID="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   RERANKER_MODEL_REVISION="233902d25c440f23af6f7d6e94d2946bac0bee0a")
        cls.proc = subprocess.Popen(["uv", "run", "--locked", "--script", "server.py"], cwd=ROOT, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(cls.url + "/health", timeout=2):
                    return
            except OSError:
                if cls.proc.poll() is not None:
                    raise RuntimeError("server.py exited during startup")
                time.sleep(1)
        cls.proc.terminate()
        raise RuntimeError("server.py did not start within 600 s")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=30)

    def call(self, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_info(self):
        status, info = self.call("/info")
        self.assertEqual(status, 200)
        self.assertEqual(info["model_id"], "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertIn(info["device"], ("mps", "cpu"))
        self.assertEqual(info["dtype"], "float32")

    def test_health(self):
        self.assertEqual(self.call("/health"), (200, {"status": "ok"}))

    def test_rerank_orders_best_first(self):
        texts = ["Lunch is served at noon.", "The reranker listens on port 8081.", "It rained on Tuesday."]
        status, ranked = self.call("/rerank", {"query": "which port does the reranker listen on?", "texts": texts})
        self.assertEqual(status, 200)
        self.assertEqual(sorted(r["index"] for r in ranked), [0, 1, 2])
        self.assertEqual(ranked[0]["index"], 1)
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_are_logits(self):
        _, ranked = self.call("/rerank", {"query": "port", "texts": ["Lunch is served at noon."]})
        self.assertLess(ranked[0]["score"], 0)  # a sigmoid would keep it in (0, 1)

    def test_empty_texts(self):
        self.assertEqual(self.call("/rerank", {"query": "anything", "texts": []}), (200, []))

    def test_bad_requests(self):
        self.assertEqual(self.call("/rerank", raw=b"not json")[0], 400)
        self.assertEqual(self.call("/rerank", {"texts": ["a"]})[0], 400)
        self.assertEqual(self.call("/rerank", {"query": 1, "texts": ["a"]})[0], 400)
        self.assertEqual(self.call("/rerank", {"query": "q", "texts": "a"})[0], 400)

    def test_unknown_paths(self):
        self.assertEqual(self.call("/nope")[0], 404)
        self.assertEqual(self.call("/nope", {"x": 1})[0], 404)


if __name__ == "__main__":
    unittest.main()
