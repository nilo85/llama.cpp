# Host info — `niklas-pc` (2× Intel Arc Pro B70)

Cheat sheet for an agent picking up the `us_*.md` / `research.md` work in this
directory. **Base facts were verified on the host on 2026-09-30 and updated on 2026-10-01.** Anything
marked ⚠️ is a trap that already cost time — read it before you build.

This file is deliberately **independent of the NixOS config**. The serving stack
in `/etc/nixos/.../vllm.nix` is the *production* path, not the dev path. You do
not need it, and you should not need to touch it.

---

## 1. Hardware

| Component | Value |
|---|---|
| CPU | **Intel Core Ultra 7 265K** — 20 cores, 1 socket, 1 NUMA node |
| P-cores | **cpu0–cpu7** (8 cores, 5.4–5.5 GHz) |
| E-cores | **cpu8–cpu19** (12 cores, 4.6 GHz) |
| Caches | L1d 704 KiB (18 inst), L2 36 MiB (11 inst), L3 30 MiB |
| RAM | **62 GiB total** (~6 GiB free, ~51 GiB in page cache at rest) |
| GPU 0 | Intel Arc Pro **B70**, `0000:04:00.0`, `/dev/dri/renderD129` |
| GPU 1 | Intel Arc Pro **B70**, `0000:84:00.0`, `/dev/dri/renderD130` |
| iGPU | `0000:00:02.0` → `card1`, driver `i915` — **do not use for offload** (see §7) |
| Storage | **Samsung SSD 990 PRO 2TB** (`nvme0n1`, non-rotational) |
| Kernel | `7.2.7` |
| OS | **NixOS 26.05 "Yarara"**, `nix 2.34.8`, flakes enabled |

GPU identity is confirmed two ways: DRM sysfs (`card0` and `card2`, driver `xe`)
and raw PCI config space, where **both cards read `8086:e223`** (Battlemage B70).

### ⚠️ CPU thread count is a known open item

`vllm.nix` currently sets `--threads 16 --threads-batch 16`. On this hybrid CPU
that very likely oversubscribes E-cores, and `research.md` §5/§6 explicitly
recommends **P-cores only** for GGML math (citing Strata #142: E-cores are slow
at expert math). **`-t 8` is an untested but well-motivated A/B.** Kernel-side
CPU placement will not fix it — llama.cpp spawns its own threads and ignores
`cpuset` affinity hints unless you use `taskset` on the whole process.

### ⚠️ PCIe link width is UNKNOWN

`research.md` claims GPU1 is on PCIe 4.0 via the southbridge (~x4) and
recommends `--tensor-split ~1.15,1`. **This could not be verified on this host:**

- `lspci` is **not installed** (no `pciutils` in the system).
- sysfs `current_link_speed` / `current_link_width` report **2.5 GT/s x1 for
  both cards** — impossible for a PCIe 5.0 x16 part. These are `xe` driver
  placeholders, not real negotiated values.
- The `xe` driver **masks `vendor_id` / `device_id`** in sysfs entirely (the
  files do not exist).
- Walking the PCIe capability (0x10) in raw config space confirms the device IDs
  above, but LNKSTA/LNKCAP also read back 2.5 GT/s x1.

**So: do not treat the 50/50 split as validated, and do not treat 1.15,1 as
established.** Measure split ratios empirically instead (§5). If you want the
real link width, install `pciutils` or read the board schematic.

---

## 2. Where things are

| What | Path |
|---|---|
| **llama.cpp repo** | `/home/niklas/workspace/llama.cpp` |
| Branch / HEAD | `niklas_fresh-master` @ `6a2743f02` ("CUDA: bitonic argsort handles rows wider than one block (#28957)") |
| Research + task docs | `/home/niklas/b70_opt/` (this dir) |
| Performance task doc | `/home/niklas/workspace/llama.cpp/docs/sycl-b70-2gpu-perf-tasks.md` |
| Container build script | `/home/niklas/workspace/llama.cpp/build-podman.sh` |
| Model cache | `/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/` |
| Production config (NixOS) | `/etc/nixos/nixos-configuration-home/machines/niklas-pc/vllm.nix` |
| Bench harnesses | `/tmp/opencode/*.py` — **⚠️ `/tmp` is not persistent** |

### Working-tree state — do not clobber

```
 M ggml/src/ggml-sycl/ggml-sycl.cpp        # env-gated profiling instrumentation
 ?? docs/sycl-b70-2gpu-perf-tasks.md
 ?? docs/sycl-igpu-offload-problem.md
```

The `ggml-sycl.cpp` change is a **measurement tool**, not a proposed patch. It is
gated behind `GGML_SYCL_OP_PROFILE=1` and it calls `wait_and_throw()` after every
op, which serialises execution (23.6 → 14.3 t/s). It is safe to leave disabled.
If you replace it, keep the env gate.

Preserved MTP-era work you should not disturb:
`backup/niklas_mtp-pre-rebase` (branch @ `e5bc21044`), `stash@{0}`,
`/tmp/opencode/ple_stub.patch`.

### Model sizes (verified on disk)

| Quant | Size | Notes |
|---|---|---|
| `UD-Q3_K_XL` | **90.0 GB** (40.0 + 50.0 shards) | **production** |
| `UD-IQ3_XXS` | **82.0 GB** | |
| `UD-Q2_K_XL` | **78.9 GB** | **download complete** (3/3 shards) |

The production model is loaded from a single shard path with
`-hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL`; llama.cpp auto-discovers the
other shards via `-m first-split-file`.

---

## 3. The single most important fact about this workload

**A 90 GB model runs in 63.8 GiB of VRAM because the PLE / n-gram table is
lazy-streamed, not resident.**

- PLE is ~51B of the model's 125B params — a **29–36 GB single tensor**.
- `llama_model_loader::lazy_read::add` (`src/llama-model-loader.cpp:1094`)
  lazy-reads any tensor `> auto_min_size = 4 GiB` when mode is `auto`.
- `--lazy-mode` is **not** set anywhere, so **`auto` is in effect**.
- Confirmed empirically: container RSS is only **4.2 GB** with **51 GB in host
  page cache**. That is mmap streaming, not a 90 GB load.

Consequence for your work: a large share of per-token cost is **NVMe row-gather
traffic for PLE, not GPU math.** Any change that adds per-row `madvise()` +
refault is expensive. This is why PRs **#29030 / #29599 / #28136** (batched
gather + prefetch; prototype measured **>2× prefill**) rank above most other
ideas, and why `us_plecache.md` (hot-row LFU) is complementary.

Do **not** pass `--load-mode dio` or `mlock` until those PRs land; both defeat
page-cache reuse.

---

## 4. Building llama.cpp — pick your path

### Path A — native CPU build (✅ verified working, no container)

Good for: editing/compiling C++, unit tests, CPU-correctness references, and
`llama-bench` logic. It will **not** touch the GPUs.

```bash
cd /home/niklas/workspace/llama.cpp
nix shell nixpkgs#cmake nixpkgs#pkg-config nixpkgs#ninja -c bash -c '
  cmake -B /tmp/opencode/build-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
  cmake --build /tmp/opencode/build-cpu --target llama-server -j 8'
```

Verified output: `version: 0.5.0-dev (build 11271, commit 6a2743f02)`.

`cmake`, `ninja`, `pkg-config`, `clang` and the oneAPI compiler are **not** on
the system PATH — you must enter a `nix shell` first. GCC 13/15, `git`, `make`
*are* on PATH.

### Path B — native SYCL/GPU build (⚠️ partially working; needs a patch)

**Good news, verified:** a native SYCL toolchain from nixpkgs **compiles SYCL
and enumerates both B70s** with no container and no NixOS involvement:

```
$ sycl-ls
[opencl:gpu][opencl:0] Intel(R) Arc(TM) Pro B70 Graphics OpenCL 3.0 NEO  [26.31.039395]
[opencl:gpu][opencl:1] Intel(R) Arc(TM) Pro B70 Graphics OpenCL 3.0 NEO  [26.31.039395]
[opencl:gpu][opencl:2] Intel(R) Graphics OpenCL 3.0 NEO  [26.31.039395]   <- iGPU
```

Note `26.31.039395` — that satisfies the "compute-runtime ≥ 26.31" requirement in
`research.md`. The runtime is reachable over **OpenCL (NEO)**, not Level Zero;
SYCL device selection must use `opencl:` selectors, **not** `level_zero:` (a
`level_zero:0` selection aborts with *"No device of requested type available"*).

**Bad news:** the full `llama-server` SYCL link does **not** complete yet,
because of a version skew between nixpkgs' MKL and what llama.cpp master expects.

**Pinned store paths (these are real, verified; re-derive if the store is GC'd):**

| Component | Store path |
|---|---|
| DPC++ compiler (`clang++`, `sycl-ls`) | `/nix/store/zcf7jg04i5byxxg1smyhcg64ayjbdc9h-intel-llvm-unstable-2025-11-14` |
| OpenCL headers | `/nix/store/vgcfwcgsij8hdz1i099b9wjpilql3n2h-opencl-headers-2025.07.22` |
| compute-runtime / NEO ICD | `/nix/store/v009b00i2s6951wavv8r3mqmy0znilhz-intel-compute-runtime-26.18.38308.1` |
| Level Zero loader | `/nix/store/akv4awajnd4mibzh471xzs0rp2d92w4b-level-zero-1.28.5` |
| MKL (**unfree**) | `/nix/store/6xg5fnzwwcxdibmh1l1mi6z5raw71kbg-mkl-2023.1.0.46342` |
| TBB (needed by MKL) | `/nix/store/dhnkzbxag2xk81p93n8980vl0iblb2fc-onetbb-2022.3.0` |

`nix intel-llvm` has **no `icpx` wrapper** — use `clang++ -fsycl`. It needs
`-I<opencl-headers>/include` for `CL/cl.h`, or compilation fails on a missing
`CL/cl.h` even though the SYCL headers are present.

**⚠️ MKL blocker (the thing that will bite you).** `ggml/src/ggml-sycl/CMakeLists.txt:184`
links `MKL::MKL_SYCL::BLAS` — a **oneAPI 2026.x** target name. nixpkgs ships
**MKL 2023.1**, which exports `MKL::MKL_DPCPP` and only defines it when
`DPCPP_COMPILER` is set. Three things are needed:

1. `mkl` is **unfree** — you need `NIXPKGS_ALLOW_UNFREE=1` **and** `nix shell --impure`
   (the env var alone is ignored without `--impure`; `~/.config/nixpkgs/config.nix`
   does *not* work for `nix shell` on a flake ref).
2. Set `DPCPP_COMPILER ON` before `find_package(MKL)`, else the target is never created.
3. Link `MKL::MKL_DPCPP` instead of `MKL::MKL_SYCL::BLAS`.
4. MKL's config also `find_package`s **TBB**, so add `nixpkgs#tbb` and put it on
   `CMAKE_PREFIX_PATH`.

Sketch (unverified end-to-end — the *configure* stage was reached, the *build*
was not):

```bash
# in ggml/src/ggml-sycl/CMakeLists.txt, near line 183
set(DPCPP_COMPILER ON)
find_package(MKL REQUIRED)
target_link_libraries(ggml-sycl PRIVATE MKL::MKL_DPCPP)   # was MKL::MKL_SYCL::BLAS
```

```bash
cd /home/niklas/workspace/llama.cpp
env NIXPKGS_ALLOW_UNFREE=1 nix shell --impure \
  nixpkgs#cmake nixpkgs#pkg-config nixpkgs#ninja \
  nixpkgs#vulkan-loader nixpkgs#vulkan-headers nixpkgs#mkl nixpkgs#tbb -c bash -c '
  export CC=<intel-llvm>/bin/clang CXX=<intel-llvm>/bin/clang++
  export CPATH=<opencl-headers>/include
  export LIBRARY_PATH=<compute-runtime>/lib:<compute-runtime>/lib/intel-opencl
  cmake -B /tmp/opencode/build-sycl -G Ninja \
    -DGGML_SYCL=ON -DGGML_SYCL_DEVICE_ARCH=bmg -DGGML_SYCL_F16=ON \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build /tmp/opencode/build-sycl --target llama-server -j 16'
```

**Expect this to need real work.** Treat Path C as the known-good baseline.

### Path C — container build (✅ known-good; use this until Path B works)

This is what production-like benchmarking should use.

```bash
cd /home/niklas/workspace/llama.cpp
TAG=llama.cpp:master-fresh ./build-podman.sh
```

It builds `localhost/llama.cpp:master-fresh` from `.devops/intel.Dockerfile` with
`-DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg`, based on
`intel/oneapi-toolkit:2026.1.1-devel-ubuntu24.04` + Level Zero 1.28.2.

**⚠️ Production does not run your build.** The live container is pinned to the
**upstream** image:

```
image = ghcr.io/ggml-org/llama.cpp:full-intel
```

So if you benchmark `master-fresh`, you are benchmarking something production is
not serving. State which image produced any number you report.

### Path D — fast oneAPI container compile check (✅ verified 2026-10-01)

Use this when the host has no `cmake`/`ninja` on PATH and you only need to
validate C++ changes before starting a long SYCL image build. It does **not**
use the GPUs and does **not** produce a runnable SYCL binary.

```bash
podman run --rm \
  -v /home/niklas/workspace/llama.cpp:/src \
  -v /tmp/opencode/llama-build:/build \
  -w /src \
  docker.io/intel/oneapi-toolkit:2026.1.1-devel-ubuntu24.04 bash -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF
  cmake --build /build --target llama -j"$(nproc)"
'
```

Notes:
- The image tag is `2026.1.1-devel-ubuntu24.04`, not `2026.1.1-devel`.
- `source /opt/intel/oneapi/setvars.sh` can return non-zero; use `|| true`.
- This CMake tree does not use `BUILD_TESTS` / `BUILD_EXAMPLES` / `BUILD_TOOLS`;
  omit those variables unless you check the current option names first.
- For a faster single-file syntax check, run:
  `icpx -fsyntax-only -std=c++17 -I include -I src -I ggml/include <file.cpp>`
  inside the same container.
- Use Path C for a standalone image, but do **not** use Path C for normal
  iteration between code changes.

### Path E — fast SYCL iteration from a mounted build dir (use this for code changes)

`build-podman.sh` bakes the binaries into a new image. That is wasteful when
iterating. For run-between-changes work, use an existing SYCL-capable image only
as the toolchain/runtime container, mount the source tree and a persistent build
dir, and rebuild/run the binaries from the mounted build dir.

For the `us-otgen-expert-ot` branch, use:

```bash
BUILD_DIR=/home/niklas/sycl-build-otgen-expert
mkdir -p "$BUILD_DIR"

# configure + incremental build
sudo podman run --rm --entrypoint bash \
  -v /home/niklas/workspace/llama.cpp:/src \
  -v "$BUILD_DIR":/build \
  -w /src \
  localhost/llama.cpp:us-otgen-expert-ot \
  -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg \
    -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF
  cmake --build /build --target llama-cli -j"$(nproc)"
'
```

Run the mounted binary with GPUs:

```bash
sudo podman run --rm --entrypoint bash \
  -e ZES_ENABLE_SYSMAN=1 \
  -v /home/niklas/workspace/llama.cpp:/src \
  -v /home/niklas/sycl-build-otgen-expert:/build \
  -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub:Z \
  --device=/dev/dri/renderD129 --device=/dev/dri/renderD130 \
  localhost/llama.cpp:us-otgen-expert-ot \
  -lc '/build/bin/llama-cli ...'
```

Notes:
- The llama.cpp podman image has `ENTRYPOINT ["/app/tools.sh"]`, so mounted-build
  iteration must use `--entrypoint bash`; otherwise `bash` is parsed as a
  `tools.sh` argument.
- `-hf` may print `HTTPS is not supported` in these container builds. If the
  model is already cached, it can still load, but for cleaner tests use the
  direct cached GGUF path under
  `/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/.../UD-Q3_K_XL/...`.
- The first mounted SYCL build is still a full build; only **subsequent**
  changes are fast incremental builds.
- The image’s `/app` binaries may be stale; that is fine. Use the image for
  oneAPI + NEO/Level Zero runtime libraries, and run `/build/bin/llama-cli`.
- Use Path C only when a standalone image is actually needed.

---

## 5. Running it

### The exact production invocation

Captured from the live process (`/proc/<pid>/cmdline`), not from the Nix config:

```
./llama-server
  -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL
  -cram 16384 --ctx-size 131072 --port 8008 --host 0.0.0.0
  --batch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0
  --flash-attn on --load-mode mmap --n-gpu-layers 48
  --threads 16 --threads-batch 16
  --split-mode layer --tensor-split 50,50 --jinja
  --ubatch-size 1024 --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

Devices: `--device=/dev/dri/renderD129 --device=/dev/dri/renderD130`
Cache mount: `-v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub:Z`

Container env that matters: **`ZES_ENABLE_SYSMAN=1`**, plus the oneAPI root vars
(`ONEAPI_ROOT`, `MKLROOT`, `TBBROOT`, `OCL_ICD_FILENAMES`, ...).

**Production service:** `llama-cpp-qwen3.8-flash-next`, listening on **:8008**,
health at `http://localhost:8008/health`. Routing in front of it is `llama-swap`
on **:8000**.

### Bench / A-B without touching production

Use a **second container on a different port**. Never restart the production
container to run an experiment.

```bash
sudo podman run -d --name ab-test --cgroups=enabled --log-driver=journald \
  -e ZES_ENABLE_SYSMAN=1 -e HF_TOKEN=PLACEHOLDER -p 8009:8009 \
  -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub:Z \
  --device=/dev/dri/renderD129 --device=/dev/dri/renderD130 \
  --pull missing ghcr.io/ggml-org/llama.cpp:full-intel \
  --server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL \
  --ctx-size 65536 --port 8009 --batch-size 2048 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --load-mode mmap \
  --n-gpu-layers 48 --split-mode layer --tensor-split 50,50
```

⚠️ **Both B70s are normally in production use.** The old "use only GPU 1 / never
touch GPU 0" rule is superseded: for dedicated GPU tests, stop
`podman-llama-cpp-qwen3.8-27b.service` with `sudo systemctl`, verify
`litellm-failover.service` remains active, run the test on both GPUs, then restart
the 27B service and check `:8006/health`. A test container that does not stop the
27B will contend for VRAM and GPUs. Stopping the 27B also frees host RAM held by
its CRAM/KV/compute buffers, so re-check `free -h` in the stopped state before
judging RAM-heavy tests. If you see OOM or
`UR_RESULT_ERROR_OUT_OF_RESOURCES`, suspect contention before suspecting your
flags. `--fit` is also unusable when `-ngl` is set: it aborts with *"failed to fit
params to free device memory: n_gpu_layers already set by user to 48, abort"*, so
**always pass `-ngl 48` explicitly** and size things yourself. For qwen4exp MTP
shared-draft runs, use `--fit off` because fitting cannot measure the draft
(`qwen4exp requires ctx_other to be set`). (Older log lines show `to 99`; the
message names whatever you passed.)

**Always clean up test containers afterwards** — they will otherwise sit on
~60 GB of VRAM and break production:
`sudo podman rm -f ab-test`

### Known-good flags (measured, Q3_K_XL, production image)

| Setting | Value | Note |
|---|---|---|
| Split mode | `layer` | `row` silently no-ops — `ggml-sycl` has no `ggml_backend_split_buffer_type` |
| Context | 131072 | 163840 OOMs; IQ3 reaches 262144 with explicit `-ngl 48` |
| KV | `q4_0` | `q8_0` now tested (2026-10-01): tg is KV-quant-insensitive here (decode not KV-bound), so this is a quality lever, not speed; q4_0 does NOT rescue 64K MTP |
| Batch / ubatch | 2048 / 1024 | research suggests 512 at long ctx; untested |
| Flash attn | `on` | keep |
| Load mode | `mmap` | never `dio`/`mlock` (see §3) |

Reference throughput (single stream, corrected timing): **23.7 t/s @500 tok,
23.5 @4k, 21.5 @16k, 14.4 @64k, 10.5 @120k**; prefill ~400–450 t/s.
Four-way concurrency only pays below ~1–2k prompt tokens; above ~8k it is slower
than one stream.

**llama-bench gotcha for Qwen3.8-Flash-Next:** `llama-bench` has no `-c` here and
can allocate the model's native 262K context, causing
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on the 2x B70 rig. For both-GPU tg
baselines, use `llama-cli` with bounded `-c` (for example `-c 4096 -n 256`) and
`--fit off`.

**Qwen3.8-Flash-Next PLE/lazy test gotchas:**
- Single-GPU `-ngl 999` OOMs for Q2_K_XL/Q3_K_XL PLE models on this rig; use both
  B70s with `-ngl 48 --split-mode layer --tensor-split 50,50 --fit off`.
- A repeated sentence prompt makes the PLE row set small/warm, so lazy-gather A/B
  shows no gain. Use a random/unique 8K-token prompt and drop caches before each
  run to measure first-run pp.
- For lazy/resident parity, disable reasoning (`--reasoning off`) and compare
  generated content only; the loading spinner and timing lines are not part of the
  output parity check.
- Long greedy `llama-cli` runs are not byte-stable on this rig for
  Qwen3.8-Flash-Next. At `-n 512 --ignore-eos`, master, `us-29245-28243-mtp-xmx`,
  and `us-29030-29245-28243` can diverge across identical reps on Q2/Q3 even with
  `--temp 0 -s 0`. Use short prefixes or a same-process parity method for
  correctness checks.
- The 3-way combo branch `us-29030-29245-28243` builds in
  `/home/niklas/sycl-build-combo` and is pushed to `nilo85`. It contains lazy PLE
  gather, grouped-MoE XMX, and MTP shared-head support.
- Podman `:ro` bind mounts do not hurt mmap/lazy reads on this rig. The model
  cache is host ext4; containers share the kernel/page cache, and llama.cpp uses
  read-only mmap plus buffered `pread` for lazy row reads.
- **MTP (`draft-mtp`) is blocked at ~64K context on this 2x32GB rig, at BOTH KV
  quants.** The draft opens a second context whose KV defaults to f16. At q8_0 it
  OOMs in `llama_kv_cache::clear` at the `seq_rm` probe (target KV 446 MiB/GPU +
  draft KV 56 MiB/GPU + draft compute buffers + 90GB Q3_K_XL weights > 2x32GB at
  70K), even forcing `-ctkd/-ctvd q8_0` and the minimum fitting `-c 71680`. At
  q4_0 (production's quant) it loads but dies during processing with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` at `ggml-sycl/cpy.cpp:1426` (cross-GPU copy),
  so q4_0 does NOT rescue 64K MTP. MTP is viable to ~32K (combo Q3: 1.52x q8_0 /
  1.53x q4_0 tg, 86-88% accept), not 64K, without a larger-VRAM rig or work on the
  cross-device copy / draft-placement path. Note tg is KV-quant-insensitive here
  (decode is not KV-bandwidth-bound), so KV quant is a quality knob, not speed/fit.
 - Dual-GPU `--split-mode layer` is pipeline-serial for single-stream tg. A
   Qwen3.6-35B-A3B Q4_K_XL control gave single-GPU tg128 `83.07 t/s` vs both-GPU
   layer `50,50` tg128 `83.24`/`83.14 t/s`. Do not expect layer split to speed
   up a single stream; it increases model capacity.
 - The two GPUs' power **flips** (one ramps to ~99% util/~240W while the other
   drops to ~0%) during prefill - this is `us_lsync`'s finding. It is INHERENT to
   the block split (GPU1's layer block needs GPU0's output each forward pass, and
   llama.cpp batches all sequences per layer, so the two blocks cannot overlap).
   It is cosmetic, not a throughput bug: concurrent decode still scales ~linearly
   (`-np 4` ~4x aggregate t/s; decode is memory-bound). `intel_gpu_top` cannot
   see the Arc B70s (`xe` driver, no i915 PMU); use `sudo nvtop -s`/`-l` (JSON
   `power_draw`/`gpu_util`) for per-GPU power. B70 #0 = `card0`/`renderD129`,
   B70 #1 = `card2`/`renderD130`, iGPU = `card1`/`renderD128`.

---

## 6. Gotchas that will waste your time

1. **`/tmp/opencode` is not persistent.** All the bench harnesses
   (`bench.py`, `bench_conc.py`, `gpu_duty.py`, `send1.py`, `run_ab.sh`,
   `syclenv.sh`, `build-cpu/`) live there. Copy them somewhere durable before
   rebooting, or regenerate them.
2. **`lspci` is not installed**, and sysfs link attributes on `xe` are
   placeholders. See §1.
3. **`vendor_id`/`device_id` are not readable** in sysfs. Use raw config space
   (`/sys/bus/pci/devices/<bdf>/config`) or DRM uevent instead.
4. **`-t 16` vs 8 P-cores** — likely suboptimal, untested. See §1.
5. **MKL version skew** breaks the native SYCL build. See §4 Path B.
6. **`nix shell` with unfree `mkl` needs `--impure` *and* `NIXPKGS_ALLOW_UNFREE=1`.**
7. **`--fit` is unusable** when `-ngl` is set; size explicitly.
8. **Both GPUs are shared with production unless you stop the 27B service.**
   Use `sudo systemctl stop podman-llama-cpp-qwen3.8-27b.service` for dedicated
   GPU tests, never stop `litellm-failover.service`, and restart the 27B after
   the test. Do not leave the 27B service stopped during docs/code review or any
   non-GPU work; restart it as soon as the GPU test window ends.
9. **nixpkgs `intel-llvm` has no `icpx`** and needs OpenCL headers for `CL/cl.h`.
10. **Do not use `--split-mode row`** on SYCL (silent no-op).
11. **Do not use the iGPU** (`0000:00:02.0`, `card1`) for offload — see §7.
12. **Production ≠ your build** (upstream `full-intel` vs `master-fresh`).
13. **`-np`/`--parallel` is not plumbed** through the Nix module. Multi-slot
    experiments need a hand-rolled container command, and the "single-slot"
    baseline in the docs is really *one active request against a 4-slot server*
    (`n_slots = 4`, `kv_unified = true`) — not a true `n_slots=1` server.

---

## 7. Dead ends already established — do not redo

- **`GGML_SYCL_PEER_MAX_BATCH_SIZE`**: 128 → 2048 changed nothing
  (prefill 400.7 → 399.7 t/s, decode 23.34 → 23.42 t/s). Noise.
- **MTP / speculative decoding**: old production-image test was acceptance 0.588,
  mean accepted length 3.92, **no speedup** - rejected for that production image.
  2026-09-30 retest on the `us-28243 + us-29245` SYCL branch (Q2_K_XL, both GPUs,
  `n_max=2`) gave **1.116x** tg (27.5 -> 30.7 t/s) and **64.7%** acceptance, so MTP
  is a net gain but below the 1.3x / 66% criterion-4 target at that quant/setting.
  3-way combo warm 512-token A/B at 8K gives Q2 **1.24x** / **81.9%** and Q3
  **1.26x** / **79.1%** (still below 1.3x). **2026-10-01: at 32K on a realistic
  prompt the 3-way combo Q3_K_XL clears BOTH criterion-4 targets — 1.52x tg q8_0 /
  1.53x q4_0 (18.44/18.28 -> 27.98/28.02 t/s) / 86-88% accept** (first pass; part of
  the gain is prompt realism, not depth alone). Two caveats remain: long greedy
  output parity across separate runs is not byte-stable (method limitation, not a
  code defect), and **64K MTP is blocked on this 2x32GB rig at BOTH KV quants**
  (q8_0 VRAM OOM, q4_0 cross-GPU copy `cpy.cpp:1426`; see the PLE/lazy gotchas).
- **`--split-mode row`**: unsupported.
- **iGPU offload**: causes GP faults. Notes in
  `docs/sycl-igpu-offload-problem.md`.
- **PLE as a *conv* optimisation target**: `ple.layers = [1]` — it is one layer
  (`src/models/qwen4exp.cpp:400-402`). The interesting PLE cost is the
  **embedding table streaming**, not the convolution (§3).
- **"Decode is at the weight-bandwidth roofline"**: false. Active-expert traffic
  is ~5.4 GB/token ≈ 127 GB/s at 23.6 t/s, roughly 7% of an assumed ~1.8 TB/s.
- **Weight traffic as the decode bottleneck**: at 23.6 t/s the GPUs average only
  ~109 W / ~99 W against a 230 W cap, and a 200 Hz duty sample found both GPUs
  busy simultaneously 54.7% of the time (strict one-at-a-time: **0.3%**). The
  GPUs are **not** simply serialising. Whatever the cross-GPU cost is, it is not
  naive serialisation.
- **Naive `perf`/`sysman` profiling**: unavailable. The per-op attribution in
  `docs/sycl-b70-2gpu-perf-tasks.md` §3.1 comes from a serialising
  `GGML_SYCL_OP_PROFILE` harness and is **ranking data only, not device timing**.

---

## 8. First things to try

1. Copy `/tmp/opencode` somewhere durable.
2. Native CPU build (Path A) to confirm you can edit + compile.
3. Container build (Path C) and A/B on port 8009 for anything performance-related.
4. Before writing kernel code, read `research.md` §10 — several of the biggest
   wins are **existing upstream PRs with existing owners**
   (#29030/#29599/#28136 PLE gather, #29245 grouped-MoE XMX GEMM, #28243 MTP).
   `AGENTS.md` in the repo asks that existing owners be contacted before forking.
