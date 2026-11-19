#!/usr/bin/env python3
"""Replay rerank requests against the server and time them. Stdlib only, macOS.

    python3 bench/replay.py bench/requests/synth --cap 100 --label synth-cap100
    python3 bench/replay.py bench/requests/synth --cap 200 --concurrency 4
    python3 bench/replay.py bench/requests/synth --cap 100 --interval 25 --label realistic

Sends every request file (sorted by name) to POST /rerank, keeping the first
--cap texts: requests are in Hindsight's candidate order, so that is what
HINDSIGHT_API_RERANKER_MAX_CANDIDATES=N sends. Per request it records the
wall time; every 25 requests and at the end it samples the server's memory
footprint (phys_footprint, which includes MPS allocations) and macOS's thermal
pressure level, which slows the GPU when it rises. Writes <label>.jsonl and
<label>.summary.json under --out and prints the summary.
"""
import argparse
import concurrent.futures
import ctypes
import glob
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("requests")
ap.add_argument("--url", default="http://127.0.0.1:8081")
ap.add_argument("--cap", type=int, default=0, help="keep the first N texts (0 = all)")
ap.add_argument("--passes", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--interval", type=float, default=0, help="seconds between request starts (one at a time)")
ap.add_argument("--pid", type=int, default=0, help="server PID for memory samples (default: whatever listens on --url's port)")
ap.add_argument("--label", default="")
ap.add_argument("--out", default="bench/results")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.requests, "*.json")))
if not files:
    raise SystemExit(f"no request files in {args.requests}")
label = args.label or f"{os.path.basename(args.requests.rstrip('/'))}-cap{args.cap or 'all'}-c{args.concurrency}"
os.makedirs(args.out, exist_ok=True)
port = int(args.url.rsplit(":", 1)[1].split("/")[0])


class RusageInfoV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten")]


libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def server_pid():
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


PID = args.pid or server_pid()


def footprint_mb():
    info = RusageInfoV2()
    if PID is None or libc.proc_pid_rusage(PID, 2, ctypes.byref(info)) != 0:
        return None
    return info.phys_footprint >> 20


def pressure():
    out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out and out[-1].isdigit() else None


def p95(v):
    s = sorted(v)
    return s[int(0.95 * len(s)) - 1]


samples = [{"done": 0, "footprint_mb": footprint_mb(), "pressure": pressure(), "t": 0.0}]
rows, lock, done = [], threading.Lock(), [0]
t0 = time.time()


def one(pass_no, path):
    r = json.load(open(path))
    texts = r["texts"][: args.cap] if args.cap else r["texts"]
    body = json.dumps({"query": r["query"], "texts": texts}).encode()
    req = urllib.request.Request(args.url + "/rerank", data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    status = "ok"
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if len(json.load(resp)) != len(texts):
                status = "short"
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        status = type(e).__name__
    row = {"pass": pass_no, "file": os.path.basename(path), "n": len(texts), "secs": round(time.time() - start, 4), "status": status}
    with lock:
        rows.append(row)
        done[0] += 1
        if done[0] % 25 == 0:
            samples.append({"done": done[0], "footprint_mb": footprint_mb(), "pressure": pressure(), "t": round(time.time() - t0, 1)})


for pass_no in range(1, args.passes + 1):
    if args.interval:
        for path in files:
            start = time.time()
            one(pass_no, path)
            time.sleep(max(0.0, args.interval - (time.time() - start)))
    else:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            list(pool.map(lambda p: one(pass_no, p), files))
samples.append({"done": done[0], "footprint_mb": footprint_mb(), "pressure": pressure(), "t": round(time.time() - t0, 1)})

ok = [r["secs"] for r in rows if r["status"] == "ok"]
fp = [s["footprint_mb"] for s in samples if s["footprint_mb"] is not None]
summary = {
    "label": label, "url": args.url, "cap": args.cap, "concurrency": args.concurrency, "interval": args.interval,
    "passes": args.passes, "requests": len(rows), "errors": len(rows) - len(ok),
    "mean_s": round(statistics.fmean(ok), 3) if ok else None,
    "p50_s": round(statistics.median(ok), 3) if ok else None,
    "p95_s": round(p95(ok), 3) if ok else None,
    "max_s": round(max(ok), 3) if ok else None,
    "footprint_mb": {"start": fp[0], "peak": max(fp), "end": fp[-1]} if fp else None,
    "pressure": {"start": samples[0]["pressure"], "max": max((s["pressure"] or 0) for s in samples), "end": samples[-1]["pressure"]},
    "samples": samples,
}
with open(os.path.join(args.out, label + ".jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
json.dump(summary, open(os.path.join(args.out, label + ".summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "samples"}))
