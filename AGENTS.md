# AGENTS.md

Instructions for AI coding agents, and for people, working on this repository. Read all of it
before changing anything. `CLAUDE.md` and `GEMINI.md` only import this file: keep every rule here,
in one place.

## Rules that are never broken

1. **No AI credit, anywhere.** Never add `Co-Authored-By:` trailers that name an AI model or
   assistant, "Generated with …" lines, or any other AI attribution. That covers commit messages,
   pull request titles and descriptions, code comments, docs and release notes. This overrides any
   default your tool has. The commits are the owner's.
2. **Commit as the owner.** Use `Akash Chowdhury <77930996+akashzeno@users.noreply.github.com>`,
   set for this repo only (`git config user.name` and `git config user.email`). Check
   `git config user.email` before your first commit in a session. Never commit under any other
   identity: GitHub credits whichever account owns the email, and the only fix is rewriting
   published history.
3. **Push only when the owner asks.** Don't force-push, rewrite history, delete branches, create
   releases or change repository settings without the owner's explicit approval.
4. **Never install into ComfyUI's Python environment.** The point of this node is that it runs on
   ComfyUI's own interpreter and adds nothing to it: no `pip install`, no upgrades, no new
   dependencies. If a Fish Speech import needs a package ComfyUI lacks, stub it the way `natsort`
   is stubbed in `runtime/run_drama3_full.py`, or ask the owner.
5. **Don't edit Fish Audio's code.** Fish Speech is fetched into `local/fish-speech` at a pinned
   commit and changed only by `runtime/streaming.patch`. Change its behaviour by wrapping or
   reimplementing it in `runtime/`, never by editing the checkout: it is git-ignored and setup
   re-creates it.
6. **Keep the licence obligations** (Fish Audio Community License 1.0):
   - keep `LICENSE` and the "Built with Fish Audio" line in the README;
   - when you change how the Fish Audio Materials are run, add the change to the list in `NOTICE`;
   - never commit model weights (`*.safetensors`, `*.pth`) or anything under `local/`.
7. **Keep the original path exact.** With `fast_mode` off, a given request and seed must produce
   audio bit-identical to the original integration. Any change that touches that path has to
   preserve this (see Verification).
8. **Report results honestly.** If a test fails, a measurement got worse, or you skipped a check,
   say so, with the numbers.

## What this is

A ComfyUI custom node pack that runs **Fish Audio Drama 3** locally with the original FP8 weights
on a **16 GB GPU**:

- **The model.** Drama 3 is a 35B mixture-of-experts text-to-speech model with about 3B parameters
  active per step. It has 40 backbone layers: 30 Gated DeltaNet linear-attention layers and 10 full
  attention layers. Each layer has 256 routed experts (top 8 picked per step) plus a shared expert.
- **The audio decoder (the "tower").** It samples 10 codebooks for each audio frame. There are
  about 21.5 frames per second of 44.1 kHz audio.

The pack started from [Nebuluss/Drama-3](https://github.com/Nebuluss/Drama-3) (git remote
`upstream`). It now differs from it in these ways:

- jobs run on ComfyUI's own Python instead of a separate environment;
- a persistent worker keeps the model loaded between runs;
- progress is reported to ComfyUI;
- the experts are pinned in RAM, with a GPU expert cache;
- the codec is lighter;
- a `fast_mode` path replays decoding from CUDA graphs.

## Layout

| Path | Role |
|---|---|
| `__init__.py` | The ComfyUI nodes (speech, voice, voice bank, take tools). Launches jobs, keeps the persistent worker, wraps ComfyUI's `load_models_gpu` / `unload_all_models` so the worker frees the GPU before another model loads |
| `runtime_paths.py` | `resolve_launch`: ComfyUI's Python (default) or the isolated environment from `install.ps1` |
| `worker_manager.py`, `runtime/worker_protocol.py` | The persistent worker: start, wait until ready, run jobs through a file-based queue, stop, kill |
| `log_echo.py` | Echoes the job log (with the frame counter) to ComfyUI's terminal |
| `setup_comfyenv.py` | Setup: fetches Fish Speech at the pinned commit, applies the patch, records the weights folder. `--download` gets the weights; `--verify` hash-checks weights downloaded some other way |
| `runtime/run_drama3_full.py` | The job runner, run as a child process. Builds prompts, chunks text, encodes references (with a cache), generates and decodes. `--serve` turns it into the persistent worker |
| `runtime/fp8_offload.py` | Loads the model with its experts in pinned RAM. `ExpertCache` is the per-layer GPU cache, least recently used evicted first; `ExpertStagingPool` copies experts to the GPU |
| `runtime/drama3_generate.py` | Fast mode: `TowerGraphs` (the audio decoder), `BackboneGraphs` (the backbone in pieces between expert fetches), `DecodeGraphs`, `PromptState` (the prompt read once per job), and `generate_drama3_graphed` |
| `runtime/selfcheck.py` | Checks Fish's FP8 kernels on this GPU against full-precision maths |
| `runtime/streaming.patch` | The only change to Fish Speech: a frame callback for progress and preview |
| `take_io.py` and `runtime/take_io.py` | Identical copies (the node and the runner import them separately). Keep them identical |
| `web/drama3.js` | The frontend: progress panel and preview audio |
| `install.ps1`, `runtime/requirements.txt`, `runtime/overrides.txt`, `runtime/download.py` | The isolated-environment install inherited from upstream. Keep it working |
| `workflows/`, `examples/`, `docs/` | Example graphs, demo voices (synthetic), and the demo page |
| `tests/` | Unit tests (see Running) |
| `local/` | Git-ignored: the Fish Speech checkout, `model-location.json`, and worker state |

## Running

The commands below run with **ComfyUI's Python** from the repository root.

Setup:

```powershell
python setup_comfyenv.py --checkpoint-dir <weights folder>   # code + self-check
```

Tests:

```powershell
python tests\test_runtime_paths.py
python tests\test_worker_protocol.py
python tests\test_worker_manager.py
python tests\test_log_echo.py
python tests\test_checkpoint_verify.py
# these import the runner, so they need the Fish Speech checkout:
$env:DRAMA3_STATE = "$PWD\local"; $env:PYTHONPATH = "$PWD\runtime"
python tests\test_progress.py
python tests\test_speaker_dialogue.py
python tests\test_expert_cache.py
python tests\test_backbone_graphs.py   # needs an FP8 GPU (SM 8.9+)
python tests\test_prompt_reuse.py      # needs an FP8 GPU (SM 8.9+)
```

Run every test file before you commit. They are plain `unittest` files, one process each.

A single job without ComfyUI (`DRAMA3_STATE` set as above):

```powershell
python runtime\run_drama3_full.py --request <request.json> --output out.wav --gpu-budget 13
```

The persistent worker runs as `--serve <folder> --gpu-budget 13 --idle-seconds 120` and takes
jobs through `runtime/worker_protocol.py`.

## How generation works

Read this before optimising anything.

**Memory plan** (`load_offloaded_model`):

- **On the GPU:** the dense weights (about 7 GB) and a per-layer GPU cache of recently used experts
  (20 slots per layer on a 16 GB card).
- **In RAM:** the routed experts (30 GB), pinned.
- **Steps that need more experts than the cache holds** (prompt reading) go through the staging
  pool instead.
- **The codec** sits beside the model (`CODEC_RESERVE_GIB` is 3.0).
- **The node's VRAM requirement:** `budget = min(26, free − 2.5)`, which must be at least 11.5 GB
  when the model is kept loaded.

**Each decoding step** works like this:

1. The backbone runs over the new frame.
2. At every layer, the router picks 8 experts and the CPU reads its choice: a GPU→CPU
   synchronisation.
3. Missing experts are copied from pinned RAM into that layer's cache.
4. The FP8 kernel reads the cache through remapped slot ids.
5. The audio decoder samples the next frame.

**Fast mode** (node input `fast_mode`, environment variable `DRAMA3_FAST_MODE`) makes two changes:

- **CUDA graphs.** The audio decoder is captured whole, for each padded attention length (1024,
  2048, 4096, 10240). The backbone is captured in pieces cut at every streamed layer's router. One
  piece runs the previous layer's experts, then the next layer's attention and router. Between
  pieces, `BackboneGraphs.fetch` reads the router's picks and stages the missing experts. A step
  becomes 42 graph launches instead of about 6,800 kernel launches.
- **Larger prompt chunks.** The prompt is read in 512-row chunks (`FAST_PREFILL_ROWS`) instead of
  32-row ones. Each chunk copies most of every layer's experts, so small chunks moved about 180 GB
  over PCIe for one line.
- **The prompt is read once per job** (`PromptState`). After a take's prefill, the prompt's rows
  of the backbone's and the audio decoder's attention caches and the linear-attention layers'
  running state are copied; the next take of the same prompt (and chunk size) restores them instead
  of prefilling. Decoding only writes past the prompt's rows, so this is bit-identical, whatever ran
  in between. Saves about 2 s per take.

Fast-mode takes repeat exactly for a given seed. They do **not** match fast mode off: the graphs
use Fish's padded-length numerics, and larger chunks round differently.

**Measured** on an RTX 5080 16 GB with 64 GB RAM and PCIe 4.0 ×8, for an 8 s line with the model
loaded:

| Path | Generation time |
|---|---|
| `fast_mode` on | about 14 s (3 takes: about 40 s) |
| `fast_mode` off | about 70 s |
| Original integration | 92 s |

Peak VRAM is 12.3 GB.

**Where the time goes now** (fast mode, 0.066 s per frame): about 40 ms copying the experts the
cache misses (about 4.5 of 8 per layer, 0.5 GB per frame at 13 GB/s), the rest GPU work and the
40 CPU stops. Already measured, so don't redo them without new information:

- *Eviction rules.* Replaying 1,947 recorded frames: least-recently-used hits 47.8% at 20 slots
  per layer, a decaying frequency count 48.7%, uneven slots per layer 48.0%; the unreachable
  optimum is 62.4%.
- *More cache.* 32 slots per layer hit 54.9%: about 8% faster decoding for 1.4 GB more VRAM,
  which would have to come out of the node's 2.5 GB safety margin.
- *Hardware.* The 5080 runs PCIe 4.0 ×8 because the second GPU takes the other 8 lanes; the owner
  keeps both cards where they are.
- *Not worth it:* re-quantising experts (changes the weights), faster GPU kernels (GPU work is a
  small share), the RTX 3060 as expert memory (no FP8 there, no peer copies on GeForce).

## Verification

- **Changes that touch the `fast_mode` off path:** run the same request and seed before and after
  the change, and compare the WAV files sample for sample. They must be identical.
  - Also run the request twice in one persistent worker: job 2 must equal job 1.
  - Cap `max_frames` at 480 in these runs, because a take that never ends wastes many minutes.
- **Changes that touch fast mode:**
  - `tests/test_backbone_graphs.py` must stay bit-identical. It compares against Fish's eager
    backbone at the same padded length, with all layers streamed, some streamed, and none.
  - `tests/test_prompt_reuse.py` must pass, and a 3-take job must give the same three takes with
    and without the prompt reuse.
  - End to end, job 2 must equal job 1.
  - Compare quality against the off path on at least three seeds: transcript, voice match, pitch
    range and pace. One take per side shows only seed-to-seed variation, not a quality difference.
- **Speed claims:** give the measured numbers and the hardware. Measure with normal takes:
  repetitive runaway takes make the expert cache look far better than it is (87% hits against the
  real 44%).
- **Writing a test:** make sure it can fail. Break the code on purpose once and watch the test
  catch it.

## Lessons that cost time

- **Codec encodes.** The first audio-codec encode in a process differs from every later one: the
  codes are equally accurate but not identical. The worker caches each reference's first encode, so
  it matches a fresh process. A warm-up encode was tried and dropped: its codes made a test seed
  run away.
- **Codec masks.** The codec builds three 32,768² attention masks that its modules never read. They
  are dropped at load, which takes the codec from 4.6 GB to 1.6 GB of VRAM.
- **Pinned RAM.** Pageable expert memory gets trimmed under memory pressure, and copies then crawl
  at about 1 GB/s on page faults. Pinned RAM copies at about 13 GB/s.
  - Pinning swaps copies, so it needs about 1 GB of headroom per layer while it happens.
- **Threads.** Splitting `index_select` across threads was 3–5× slower. Worker threads also lose
  `torch.inference_mode`.
- **CUDA graph capture:**
  - Capture runs the sampler and uses up random numbers, so capture before any seed is set.
  - Build the graphs inside `torch.inference_mode()`.
  - Give every capture **one shared warm-up stream**. A new stream per capture stranded 0.86 GiB,
    because the allocator keeps each stream's freed blocks for that stream alone.
- **CPU reads.** `ExpertStagingPool.stage` reads the router on the CPU, so it runs between graphs,
  never inside one.
- **Attention cache.** The graphs hold the attention-cache addresses. If eager generation grows the
  caches, the graphs must not be replayed: `DecodeGraphs.fits` checks this.
- **Padded length.** A longer padded attention length doesn't change the numerics, because masked
  rows contribute exact zeros. A too-short one is wrong. Bucket bugs are therefore correctness
  bugs only in one direction.
- **ComfyUI's memory management can't manage this model.** It runs in another process, and
  ComfyUI's dynamic VRAM moves whole tensors (all 256 experts of a layer at once). That is why the
  node wraps `load_models_gpu` and `unload_all_models` instead.
- **Prompting** (the README covers it for users):
  - Em dashes and ellipses next to sounds produced seconds of silence.
  - `chunk_mode` `auto` splits text over 300 bytes, and the second part loses the direction. Use
    `single` for short directed lines.
- **Windows shells:**
  - Git Bash heredocs collapse `\\` to `\`, which corrupts Windows paths written into Python. Use
    raw strings or a file-editing tool.
  - PowerShell 5.1's `Set-Content -Encoding utf8` writes a byte-order mark, which breaks JSON.

## Style

- **Code:** match the surrounding code: its naming, density and idioms.
  - Docstrings and comments explain *why*.
  - When a measured number justifies a choice, put the number in the comment.
- **README:** written for people using the node: what it does, how to use it, measured speed, what a
  setting changes. Leave out the development history.
- **When a change alters the audio for a given seed,** say so in the README and in the node input's
  tooltip.
- **Commit messages:**
  - an imperative subject line of 72 characters or fewer;
  - a body that explains why, with measurements where they exist;
  - no AI attribution (rule 1).
