# Local LLM Failover

## The gist

opencode talks to a local proxy, and the proxy picks the backend per request:

```
opencode ──> LiteLLM  127.0.0.1:4000   (litellm-failover.service)
               ├──> qwen3.8-27b   @ llama.cpp :8006   ← primary, local GPU
               └──> Qwen3.8-Flash @ DeepInfra         ← fallback, automatic
```

Ask for `qwen3.8-27b` and you normally get the local 27B. If the local server is
unreachable, LiteLLM silently retries against DeepInfra — same model name, same
request, no intervention. After 2 consecutive failures the local backend is
cooled down for 60s so a dead endpoint isn't retried on every call.

`llama-swap` is not in this path. The proxy goes straight to the llama.cpp
server on `:8006`.

## How to use it

```
niklas-pc-failover/qwen3.8-27b               # local 27B, auto-fails-over
niklas-pc-failover/qwen3.8-flash-deepinfra   # always DeepInfra, never touches the GPU
```

The old `niklas-pc` provider (direct to llama-swap on `:8000`) still exists and
is unchanged.

## GPU access

The 27B server holds the GPU. Stop it before GPU work, restart it after.

```
sudo systemctl stop  podman-llama-cpp-qwen3.8-27b.service   # before testing
sudo intel_gpu_top                                          # confirm it's free
sudo systemctl start podman-llama-cpp-qwen3.8-27b.service   # when done
```

Stopping it does not break the session — the fallback covers you, requests
just route to DeepInfra meanwhile. That's the whole point of the setup.

**Two things to respect:**

- Do NOT stop `litellm-failover.service`. That service is the fallback.
  Stopping the 27B is expected and safe; stopping the proxy is not.
- Restarting is not instant. The 27B has to load back into VRAM, so allow
  time before concluding something is broken.

There are 8 llama container units defined. `podman-llama-cpp-qwen3.8-27b` is the
one normally running, but a session may have started others — check before
assuming the GPU is free:

```
systemctl list-units 'podman-llama-cpp*' --state=running
```

Ad-hoc `podman run` containers can also hold the GPU without appearing there.

## Gotchas

- Images degrade on failover. The local 27B is multimodal, DeepInfra is
  text-only. A request carrying an image that fails over will not come back
  intact.
- Context is pinned to 170k on both models, matching the local 27B rather than
  DeepInfra's advertised 1M. This keeps the compaction threshold identical
  whichever backend answers, so a long session can't outgrow the local model.
- The fallback is metered (~$0.113/M in, ~$0.382/M out). Fine for occasional
  outages, worth knowing if the local server stays down.
- Startup order: `litellm-failover` is not ordered after the 27B unit on
  purpose, so it starts and serves the fallback even when the local model is
  down.

Config lives in `machines/niklas-pc/llm-failover.nix`; provider definitions are
in `~/.config/opencode/opencode.jsonc`.

## Implication for b70_opt work

Because the proxy auto-fails-over to DeepInfra, **stopping the 27B server frees
ALL GPUs for testing** without breaking this session. So for any GPU-bound work
(MTP test, SYCL bench, etc.): stop `podman-llama-cpp-qwen3.8-27b.service`, use
the GPUs, then restart it. Never stop `litellm-failover.service`.
