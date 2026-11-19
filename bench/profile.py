# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "transformers==5.17.0",
# ]
# ///
"""Turn rerank request files into a length profile. No text leaves this step.

    uv run --locked --script bench/profile.py bench/requests/mine -o bench/profiles/mine.json

For every request it records the query's token count and, for every
candidate, the token count of the (query, text) pair as the model's tokenizer
encodes it, truncated at the model's limit, in the request's order. Cross-
encoder time depends on these counts and on how many candidates there are,
not on the words, so bench/synth.py can rebuild requests of the same shape.
"""
import argparse
import glob
import json
import os

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("requests", help="directory of request JSON files ({query, texts})")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--model", default="hindsight-ai/hs-rerank-1")
ap.add_argument("--revision", default="927f5be31280298a51cf29af8a29cb111ed812c5")
ap.add_argument("--note", default="", help="free-text description stored in the profile")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
reqs = []
for path in sorted(glob.glob(os.path.join(args.requests, "*.json"))):
    r = json.load(open(path))
    q, texts = r["query"], r["texts"]
    pairs = tok([q] * len(texts), texts, truncation=True)["input_ids"] if texts else []
    reqs.append({"query_tokens": len(tok(q)["input_ids"]), "pair_tokens": [len(p) for p in pairs]})

json.dump(
    {"model": args.model, "revision": args.revision, "max_length": tok.model_max_length, "note": args.note, "requests": reqs},
    open(args.out, "w"),
)
n = sum(len(r["pair_tokens"]) for r in reqs)
print(f"{len(reqs)} requests, {n} pairs -> {args.out}")
