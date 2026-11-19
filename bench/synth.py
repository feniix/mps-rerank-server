# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "transformers==5.17.0",
# ]
# ///
"""Build synthetic rerank requests that match a length profile.

    uv run --locked --script bench/synth.py bench/profiles/agent-host-2026-11.json -o bench/requests/synth

Each request gets a query and candidate texts made of pseudo-words from a fixed
seed, sized so that every (query, text) pair has the profile's token count
under the model's tokenizer. The words mean nothing; the shapes match, and
shape is what a cross-encoder's time and memory depend on. Prints the largest
deviation from the profile in tokens.
"""
import argparse
import json
import os
import random

# Only the tokenizer is needed; keep transformers from warning that PyTorch is absent.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("profile")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

prof = json.load(open(args.profile))
tok = AutoTokenizer.from_pretrained(prof["model"], revision=prof["revision"])
rnd = random.Random(args.seed)
SYL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def words(n):
    return " ".join("".join(rnd.choice(SYL) for _ in range(rnd.randint(1, 3))) for _ in range(n))


def text_of(n_tokens):
    """Text that encodes to exactly n_tokens content tokens (no special tokens)."""
    if n_tokens <= 0:
        return ""
    s = words(n_tokens + 8)
    while len(tok(s, add_special_tokens=False)["input_ids"]) < n_tokens:
        s += " " + words(n_tokens)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    s = s[: enc["offset_mapping"][n_tokens - 1][1]]
    # Cutting can merge or split the last word; nudge until the count matches.
    for _ in range(20):
        k = len(tok(s, add_special_tokens=False)["input_ids"])
        if k == n_tokens:
            break
        s = s[:-1].rstrip() if k > n_tokens else s + " ba"
    return s


os.makedirs(args.out, exist_ok=True)
worst = 0
for n, r in enumerate(prof["requests"], 1):
    q = text_of(r["query_tokens"] - len(tok("")["input_ids"]))
    overhead = len(tok(q, "")["input_ids"])  # query, special tokens, empty text
    texts = []
    for p in r["pair_tokens"]:
        t = text_of(p - overhead) if p < prof["max_length"] else text_of(prof["max_length"])
        got = len(tok(q, t, truncation=True)["input_ids"])
        worst = max(worst, abs(got - p))
        texts.append(t)
    json.dump({"query": q, "texts": texts}, open(os.path.join(args.out, f"req-{n:05d}.json"), "w"))
print(f"{len(prof['requests'])} requests -> {args.out}; largest pair-length deviation {worst} tokens")
