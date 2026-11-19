# mps-rerank-server

A small HTTP server that runs a cross-encoder reranker on an Apple Silicon
GPU (PyTorch MPS) behind the API of
[Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
(TEI). It was written for [Hindsight](https://github.com/vectorize-io/hindsight)'s
`tei` reranker provider, and serves Hindsight's default reranker,
[`hindsight-ai/hs-rerank-1`](https://huggingface.co/hindsight-ai/hs-rerank-1),
in FP16.

It is one Python file (`server.py`) with its lockfile, a launchd agent
template, three shell scripts to install, inspect and remove it, a small
benchmark kit (`bench/`) and tests.

## Why it runs on macOS, not in Docker

On a Mac, Hindsight's built-in `local` reranker runs on the CPU, whether
Hindsight runs in Docker or natively.

**In Docker.** Docker on macOS (Docker Desktop, OrbStack, Colima) runs
containers inside a Linux virtual machine. The Apple GPU is reachable only
through Metal, a macOS framework, and the VM does not pass the GPU through
to Linux, so PyTorch inside a container sees neither CUDA nor MPS and falls
back to the CPU. (Some VMs expose the GPU to Linux through Vulkan, such as
Podman with krunkit, which llama.cpp can use; PyTorch has no backend for it.)
The VM also takes a fixed slice of the Mac's memory, and a model loaded in a
container has to fit in that slice.

**Natively.** Hindsight 1.2.x never selects MPS for its local reranker. It
was disabled after a Metal crash under concurrent reranking
([vectorize-io/hindsight#4412](https://github.com/vectorize-io/hindsight/issues/4412))
and because MPS keeps a per-shape memory cache that grows with
variable-length inputs.

So the model runs here, as a macOS process next to Hindsight:

- PyTorch runs natively, reaches the GPU through Metal (MPS) and uses the
  Mac's unified memory directly.
- Hindsight, in a container or not, calls it over HTTP with its `tei`
  provider. From a container the address is `host.docker.internal`.
- The two concerns behind the upstream decision are contained in this
  process: one forward pass at a time (no concurrent Metal work), and a
  periodic cache flush. When MPS memory grows anyway, you restart this
  process, not Hindsight.

hs-rerank-1 reads up to 8,192 tokens per pair, so the CPU-to-GPU difference
is large. Hindsight's other Apple-GPU route is its in-process `jina-mlx`
provider, which uses a different model.

## API

| Call | Request | Response |
|---|---|---|
| `GET /info` | | `{"model_id", "revision", "device", "dtype"}` |
| `GET /health` | | `{"status": "ok"}` |
| `POST /rerank` | `{"query": "...", "texts": ["...", ...]}` | `[{"index": 0, "score": 1.23}, ...]`, best first |

Scores are raw logits, the same scale Hindsight's `local` provider returns;
Hindsight applies the sigmoid itself. The model's own sigmoid is replaced by
an identity so the scores are not squashed twice.

## Requirements

- An Apple Silicon Mac (M1 or later). Without MPS the server falls back to
  the CPU and logs a warning.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.13 and the pinned
  dependencies (PyTorch, sentence-transformers, transformers) from
  `server.py.lock` on first run.
- About 2 GB of disk for PyTorch and the model, and 2–4 GB of memory while
  serving (more with long texts; see Limits).

## Run it in the foreground

```sh
uv run --locked --script server.py
```

The first run downloads Python 3.13, PyTorch and the model, about 2 GB; on a
fast connection that took under a minute. Warnings about unauthenticated
Hugging Face requests and a missing `modules.json` are harmless.
When the log says `serving hindsight-ai/hs-rerank-1@927f5be3 (float16) on mps`,
try it:

```sh
curl -s localhost:8081/info
curl -s localhost:8081/rerank -H 'Content-Type: application/json' \
  -d '{"query": "which port does the reranker use?", "texts": ["The reranker listens on port 8081.", "Lunch is at noon."]}'
```

## Deploy as a launchd agent

For a machine that should always have the reranker running, install it as a
per-user launchd agent. It starts at login and restarts if it crashes.

```sh
git clone https://github.com/feniix/mps-rerank-server.git
cd mps-rerank-server
scripts/install.sh
scripts/status.sh
```

`scripts/install.sh` does the following:

1. Finds `uv` on your `PATH`. launchd starts agents with a minimal `PATH`,
   so the agent refers to uv by absolute path.
2. Renders `launchd/io.github.feniix.mps-rerank-server.plist` into
   `~/Library/LaunchAgents/`, filling in the uv path, this directory, your
   home directory, and the host and port (`RERANKER_HOST`, default
   `127.0.0.1`; `RERANKER_PORT`, default `8081`).
3. Checks the result with `plutil -lint`.
4. Stops a running copy (`launchctl bootout`) and waits until launchd has
   really removed it. Loading again too early fails with
   `Bootstrap failed: 5: Input/output error`.
5. Loads the agent (`launchctl bootstrap gui/<uid>`). With `RunAtLoad` it
   starts at once and at every login; with `KeepAlive` launchd restarts it
   after a crash, at most once every 10 seconds (`ThrottleInterval`).

The agent runs `uv run --locked --script server.py` from this directory, as
your user, with `ProcessType` set to `Interactive` so macOS does not
throttle it as a background job. Output goes to
`~/Library/Logs/mps-rerank-server.log`. The first start downloads PyTorch and
the model, so `scripts/status.sh` may report the server as not responding
for a few minutes; follow the log meanwhile:

```sh
tail -f ~/Library/Logs/mps-rerank-server.log
```

Day to day:

| Task | Command |
|---|---|
| Check state, PID, last exit code and `/info` | `scripts/status.sh` |
| Restart (also resets MPS memory, see Limits) | `scripts/install.sh` |
| Use another port | `RERANKER_PORT=8090 scripts/install.sh` |
| Stop and remove the agent | `scripts/uninstall.sh` |

Re-run `scripts/install.sh` after moving the directory or upgrading uv,
because the agent holds both paths. The agent runs from the clone itself, so
`git pull` followed by `scripts/install.sh` deploys a new version.

To run it for all users at boot, without a login, it would have to be a
LaunchDaemon in `/Library/LaunchDaemons`, with a `UserName` key and root to
install it. The scripts here only handle the per-user agent.

## Point Hindsight at it

Set the `tei` provider and its URL in Hindsight's environment.

Hindsight in Docker on the same Mac (tested with OrbStack, which routes
`host.docker.internal` to the host's loopback):

```yaml
HINDSIGHT_API_RERANKER_PROVIDER: tei
HINDSIGHT_API_RERANKER_TEI_URL: http://host.docker.internal:8081
```

Hindsight installed natively on the same Mac:

```sh
HINDSIGHT_API_RERANKER_PROVIDER=tei
HINDSIGHT_API_RERANKER_TEI_URL=http://127.0.0.1:8081
```

To keep recall working when the server is down, add a fallback member. After
the `tei` provider's retries (about 3.5 s), Hindsight uses member 1 and logs
only a warning, so a sudden slow recall is a reason to run
`scripts/status.sh`:

```yaml
HINDSIGHT_API_RERANKER_1_PROVIDER: local
HINDSIGHT_API_RERANKER_1_LOCAL_MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
```

MiniLM-L6 is a light CPU model, so the fallback is fast but ranks worse.

How many candidates Hindsight sends per recall is
`HINDSIGHT_API_RERANKER_MAX_CANDIDATES` (default 300). Rerank time grows
about linearly with it.

## Benchmark

`bench/` measures rerank latency and memory on your Mac, with your own
requests or with synthetic ones of the same shape.

| Script | What it does | Needs |
|---|---|---|
| `bench/record.py` | Proxy that saves every `/rerank` body Hindsight sends | Python 3 |
| `bench/profile.py` | Reduces saved requests to token lengths (no text) | uv |
| `bench/synth.py` | Builds fake requests that match a profile's token lengths | uv |
| `bench/replay.py` | Replays requests at a cap, concurrency or rate; records time, memory and thermal pressure | Python 3, macOS |

Cross-encoder time and memory depend on how many candidates a request has
and how many tokens each (query, text) pair has, not on what the words say.
That is why a profile of token lengths is enough to rebuild a workload
without sharing its text.

Record real requests (they contain your memory text and stay local;
`bench/requests/` is git-ignored):

```sh
python3 bench/record.py --out bench/requests/mine
# point Hindsight at the recorder for a while:
#   HINDSIGHT_API_RERANKER_TEI_URL=http://127.0.0.1:8082   (or host.docker.internal:8082)
# set HINDSIGHT_API_RERANKER_MAX_CANDIDATES to the largest cap you want to test
```

Turn them into a shareable profile, or build synthetic requests from one:

```sh
uv run --locked --script bench/profile.py bench/requests/mine -o bench/profiles/mine.json
uv run --locked --script bench/synth.py bench/profiles/mine.json -o bench/requests/synth
```

`synth.py` matches every pair's token count to within one token and prints
the largest deviation. Synthetic requests run somewhat faster than the real
ones they copy: sentence-transformers sorts pairs by character count before
batching, real text has an uneven characters-per-token ratio and so batches
with more padding, and the synthetic words have an even ratio. On one
measured workload (100 real recall queries, caps 100 and 200 on an M4 Mac
mini) synthetic p50 was 10–14% lower and p95 5–7% lower. Treat synthetic
timings as a lower bound, and measure real requests when the number
matters.

Replay against the running server:

```sh
python3 bench/replay.py bench/requests/synth --cap 100              # one at a time
python3 bench/replay.py bench/requests/synth --cap 200 --concurrency 4
python3 bench/replay.py bench/requests/synth --cap 100 --interval 25   # one request every 25 s
```

`--cap N` keeps the first N texts of each request, which is what
`HINDSIGHT_API_RERANKER_MAX_CANDIDATES=N` would have sent, so one recording
at a high cap serves every lower one. Results go to `bench/results/`:
per-request times and a summary with mean, p50, p95 and max seconds, the
server's memory footprint (including MPS allocations) and macOS's thermal
pressure level. Record the pressure with every run: when it rises, macOS
lowers the GPU clock and the same requests take longer.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

`tests/test_server.py` starts `server.py` on a free port with the small
MiniLM-L6 model and checks the API: `/info`, `/health`, ordering, raw-logit
scores, empty and malformed requests. `tests/test_synth.py` checks that
`bench/synth.py` reproduces a profile's token lengths. The first run
downloads PyTorch and the models. There is no CI: hosted macOS runners have
no usable MPS, so GPU behaviour is checked on a real Mac.

## Configuration

Environment variables read by `server.py`. For the agent, add them to the
`EnvironmentVariables` dictionary in the plist template and re-run
`scripts/install.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `RERANKER_HOST` | `127.0.0.1` | Interface to bind. The server has no authentication; keep it on loopback. |
| `RERANKER_PORT` | `8081` | Port. |
| `RERANKER_MODEL_ID` | `hindsight-ai/hs-rerank-1` | Any sentence-transformers cross-encoder on the Hugging Face Hub. |
| `RERANKER_MODEL_REVISION` | `927f5be31280298a51cf29af8a29cb111ed812c5` | Pinned so an upstream re-upload cannot change rankings. Change it with the model. |
| `RERANKER_DTYPE` | `float16` | `float16` for hs-rerank-1 (its weights ship in FP16); `float32` for models such as MiniLM-L6. |
| `RERANKER_BATCH_SIZE` | `32` | Pairs per forward pass. |
| `RERANKER_EMPTY_CACHE_SECS` | `3600` | How often to return cached MPS buffers to the driver; `0` disables it. |

## Limits

- **One request at a time.** A lock around the forward pass serializes
  requests, so concurrent recalls queue. With four callers at once, each
  waits about four times as long as a single caller.
- **MPS memory grows.** MPS keeps buffers for every input shape it has seen,
  and long, variable-length texts make that pool grow: we have seen the
  footprint go from about 2 GB to over 4 GB within an hour on long memories,
  and to 7.5 GB after about two days. The server hands cached buffers back to
  the driver every hour, but PyTorch has no call to clear the per-shape graph
  cache, so the footprint still creeps. Restarting (`scripts/install.sh`)
  resets it.
- **Thermal throttling.** Sustained GPU load on a small Mac can raise macOS's
  thermal pressure, and macOS then lowers the GPU clock, which slows
  reranking. Keep an eye on latency under long, continuous load.
- **No authentication and no TLS.** It is meant for loopback use on the
  machine that runs Hindsight.
- **macOS only.** On Linux with an NVIDIA GPU, run TEI itself or Hindsight's
  `local` provider, which uses CUDA.

## Licence

MIT, see [LICENSE](LICENSE). The model is downloaded from the Hugging Face Hub
at first start and has its own licence; check the model card.
