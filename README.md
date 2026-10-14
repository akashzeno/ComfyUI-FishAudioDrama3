# ComfyUI-FishAudioDrama3

Run **[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)** — the 35B mixture-of-experts
text-to-speech model that takes stage directions — locally in ComfyUI, with the **original FP8
weights**, on a **16 GB GPU**.

**Built with Fish Audio.** Unofficial integration; not affiliated with Fish Audio.

- **Runs in ComfyUI's own Python.** No second environment and nothing pip-installed: a setup script
  fetches Fish Speech's inference code at a pinned commit and checks the GPU kernels before you
  download anything.
- **Fits a 16 GB card.** About 7 GB of the model stays on the GPU; the routed experts (28 GB) stream
  from system RAM as each step needs them. Same weights, same maths — nothing is re-quantised.
- **Keeps the model loaded between runs.** The first run of a session loads it (about 1.5 min);
  later runs start generating at once. It hands the GPU back on its own the moment ComfyUI loads any
  other model, after an idle period, or when you press *Free memory*.
- **Shows its work.** The node's progress bar moves through loading and generation, and the model's
  own log — its frame counter included — appears in ComfyUI's terminal.
- **The whole of Drama 3:** voice cloning from a clip and its transcript, natural-language
  directions, inline sounds and word tags, up to 16 speakers per script, long scripts, several
  seeds per run, previous-take context, and live preview.

## Requirements

| | |
|---|---|
| GPU | NVIDIA with FP8 tensor cores (RTX 40 or 50 series, SM 8.9+). Tested: RTX 5080 16 GB |
| System RAM | 64 GB (the experts live there, pinned) |
| Disk | about 45 GB: 41.5 GB of weights plus Fish Speech's code |
| ComfyUI | a current build with Triton (tested: ComfyUI 0.39, torch 2.14.1, Triton 3.8, Windows) |
| Other | `git` on PATH |

## Install

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/akashzeno/ComfyUI-FishAudioDrama3.git
cd ComfyUI-FishAudioDrama3

# with ComfyUI's Python:
python setup_comfyenv.py --checkpoint-dir ..\..\models\TTS\drama-3              # code + self-check
python setup_comfyenv.py --checkpoint-dir ..\..\models\TTS\drama-3 --download   # + the weights
```

1. **The first command** fetches Fish Speech at the pinned commit into `local/` (git-ignored),
   applies a small streaming patch, records where the weights go, and runs a **self-check**: it
   imports everything a job needs, then runs Fish's FP8 kernels on your GPU against the same maths
   in full precision. If that fails, your setup can't run Drama 3 — and you have not downloaded
   41.5 GB to find out.
2. **The second** downloads the checkpoint at the pinned revision (it resumes if interrupted).
   You can also download the files yourself — `codec.pth`, the eight `model-0000N-of-00008.safetensors`
   and the small `*.json` files, from the pinned revision — and rerun the first command.
3. **Restart ComfyUI.** The nodes are under **audio → Drama 3**.

## Quick start: clone a voice and direct it

```text
Load Audio (your voice clip) ──→ Drama 3 - FP8 Speech (RAM Offload) ──→ Save Audio
                                   reference_audio + reference_text
```

- **`reference_audio`** — a clean clip of one speaker, 3–30 s.
- **`reference_text`** — exactly what is said in that clip. A wrong transcript weakens the clone.
- **`text`** — the line to speak, with its direction:

```text
[warm and relaxed, like talking to an old friend] It's been a while. [small laugh] Too long, honestly.
```

`workflows/` has ready-made graphs: cloning, multi-voice dialogue, long text, variations,
continuation, redirection, progressive preview and more. Their demo voices are in `examples/`.

## Writing the text

| Tool | Write | What it does |
|---|---|---|
| Direction | `[slow, dangerously quiet]` before the words | Brief it like an actor: emotion, intent, pace, who is being addressed. It holds until the next direction, so a second one changes the mood mid-line |
| Sound or pause | `[pause]` `[small laugh]` `[laughs]` `[sigh]` `[shaky breath]` | Placed exactly where it happens, and nowhere else |
| Word tags | `<emphasis>…</emphasis>` `<stress>` `<soft>` `<slow>` `<fast>` `<forceful>` `<whisper>` | Change only the words they wrap |
| Speakers | `<\|speaker:0\|>` … `<\|speaker:15\|>` | Several voices in one script; give each a reference (Voice Reference → Voice Bank) |
| Pronunciation | `<\|phoneme_start\|>N IY1 L<\|phoneme_end\|>` | CMU Arpabet, for names it gets wrong |

What we learned the hard way:

- **Avoid em dashes and ellipses** next to sounds. `— [small laugh]` produced a 7.6 s stretch of
  silence in one take; commas and full stops give natural pauses.
- **Put the direction in `text` and set `chunk_mode` to `single`** for short directed lines. In
  `auto` mode, text over `chunk_bytes` is split into parts, and only the first part carries the
  direction you wrote at its start.
- **Set `max_frames`.** Occasionally a take never ends. 480 frames is about 22 s; the default (0)
  allows 2,048 frames — 95 s of audio and many minutes of waiting.
- **Generate a few seeds.** Takes vary; `variations` = 3 gives seeds `seed`, `seed+1`, `seed+2` in one run.

## The speech node's settings

| Input | Default | Notes |
|---|---|---|
| `text` | — | What to say, with directions and tags |
| `seed` / `variations` | 42 / 1 | Takes use consecutive seeds |
| `max_frames` | 0 (auto) | A cap per part; ~21.5 frames per second of audio |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.7 / 30 | Lower temperature gives steadier, flatter reads |
| `direction` | empty | A direction added in front of every part — use it *or* write the direction in `text` |
| `chunk_mode` | auto | `single` keeps the text in one part; `speakers`, `lines` and `clauses` split it |
| `previous_take` | — | Continue from an earlier take or recording |
| `streaming` | off | Decode and play blocks while generating (uses more VRAM) |
| `keep_loaded` | on | Keep the model loaded between runs |
| `idle_unload_minutes` | 10 | Unload the kept model after this long without a run |
| `release_comfy_models` | on | Unload ComfyUI's models before loading Drama 3 |
| `runtime_folder` | blank | Blank: ComfyUI's Python. A folder made by `install.ps1`: that isolated environment |

## How it works

- **A separate process.** Drama 3 runs in a child process started with ComfyUI's own interpreter,
  so a 35B model that runs out of memory cannot take ComfyUI down with it. With `keep_loaded` that
  process stays alive and takes jobs one after another; otherwise it exits after each job.
- **Experts from RAM.** The attention layers, shared layers and as many expert layers as fit in your
  VRAM stay on the GPU. For the rest, each step copies only the 8 experts (of 256) the router picks,
  from pinned RAM, in their original FP8 form.
- **Giving the GPU back.** ComfyUI can't see memory used by another process, so the node wraps
  `load_models_gpu` and `unload_all_models`: before any other model loads onto the GPU, the Drama 3
  process is asked to exit.
- **Same output as a fresh process.** The first audio-codec encode in any process takes slightly
  different floating-point paths than every later one. The kept-loaded process encodes each reference
  once and reuses those codes, so its takes match a fresh process bit for bit (checked on an RTX 5080).

## Speed (RTX 5080 16 GB, 64 GB RAM, PCIe 4.0 ×8)

| | |
|---|---|
| Loading codec + model | about 1.5–2 min, once per session with `keep_loaded` |
| A 7.4 s line, model already loaded | about 105 s |
| The same line, loading per run | 203 s |
| Peak memory | about 14 GB VRAM, 30 GB RAM |

Generation runs about 12× slower than real time on this card because most experts stream from RAM;
more VRAM keeps more of them resident.

## Troubleshooting

- **"Drama 3 is not set up"** — run `setup_comfyenv.py` (see Install).
- **"Free other GPU workloads"** — the job needs about 11 GB free on the GPU. Close other GPU apps.
- **Slow, or the PC swaps** — the model pins about 28 GB of RAM; close heavy apps (browsers, editors).
- **A take keeps going** — set `max_frames`, and stop the run from ComfyUI; the process is killed.
- **Check the install** — `python setup_comfyenv.py` reruns the self-check at any time.

## Tests

```powershell
python tests\test_runtime_paths.py
python tests\test_worker_protocol.py
python tests\test_worker_manager.py
python tests\test_log_echo.py
# these import the runner, so they need the Fish Speech checkout:
$env:DRAMA3_STATE = "$PWD\local"; $env:PYTHONPATH = "$PWD\runtime"
python tests\test_progress.py
python tests\test_speaker_dialogue.py
```

## Credits and licence

- **[Nebuluss/Drama-3](https://github.com/Nebuluss/Drama-3)** — the original ComfyUI integration
  this project is built on: the RAM expert staging, the speech, voice and take nodes, the workflows
  and the examples. Its isolated-environment installer (`install.ps1`) still works here.
- **[Fish Audio](https://fish.audio)** — [Drama 3](https://huggingface.co/fishaudio/drama-3) and
  [Fish Speech](https://github.com/fishaudio/fish-speech).

The Fish Audio Materials are licensed under the **Fish Audio Community License, Version 1.0** — see
`LICENSE`, and `NOTICE` for the attribution and the list of changes. In short: research and
non-commercial use are free; commercial use is free while your annual revenue is under US$500,000;
running it as a hosted speech service for others needs a separate licence from Fish Audio; and you
may not imitate a real person's voice without their consent. Model weights are not included here.
