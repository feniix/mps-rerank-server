"""bench/synth.py reproduces a profile's token lengths. Run with the other tests.

Writes a small profile, builds synthetic requests from it, profiles them again
with bench/profile.py and compares pair by pair. Needs uv on PATH.
"""
import json
import os
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SynthMatchesProfile(unittest.TestCase):
    def test_round_trip(self):
        want = {
            "model": "hindsight-ai/hs-rerank-1", "revision": "927f5be31280298a51cf29af8a29cb111ed812c5",
            "max_length": 8192, "note": "test",
            "requests": [
                {"query_tokens": 12, "pair_tokens": [40, 95, 300, 1200]},
                {"query_tokens": 60, "pair_tokens": [70, 61 + 1, 2048, 8192]},
                {"query_tokens": 5, "pair_tokens": []},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            prof, reqs, again = os.path.join(d, "p.json"), os.path.join(d, "reqs"), os.path.join(d, "again.json")
            with open(prof, "w") as f:
                json.dump(want, f)
            run = lambda *a: subprocess.run(["uv", "run", "--locked", "--script", *a], cwd=ROOT, check=True, capture_output=True)
            run("bench/synth.py", prof, "-o", reqs)
            run("bench/profile.py", reqs, "-o", again)
            with open(again) as f:
                got = json.load(f)
        for w, g in zip(want["requests"], got["requests"]):
            self.assertEqual(g["query_tokens"], w["query_tokens"])
            self.assertEqual(len(g["pair_tokens"]), len(w["pair_tokens"]))
            for a, b in zip(g["pair_tokens"], w["pair_tokens"]):
                self.assertLessEqual(abs(a - b), 1, (a, b))


if __name__ == "__main__":
    unittest.main()
