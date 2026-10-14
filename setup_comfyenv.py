"""Set Drama 3 up to run with ComfyUI's own Python — no new environment, nothing pip-installed.

Run it with ComfyUI's Python (the comfyenv interpreter):

    python setup_comfyenv.py --checkpoint-dir D:\\Models\\drama-3            # code + self-check
    python setup_comfyenv.py --checkpoint-dir D:\\Models\\drama-3 --download # also the 41.5 GB weights

Steps, each skipped when already done:
  1. Fish Speech at the pinned commit (sparse checkout of fish_speech/ and tools/) into the state
     folder, with this repo's small streaming patch applied.
  2. model-location.json pointing at the checkpoint folder.
  3. --download: the pinned original FP8 checkpoint from fishaudio/drama-3 (runtime/download.py).
  4. Self-check (runtime/selfcheck.py): every import a job needs, then Fish's FP8 MoE kernels on
     the GPU against the same maths in fp32 — run it before downloading the weights.

The pins are the ones install.ps1 uses, so both modes run the same code and the same checkpoint.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NODE = Path(__file__).resolve().parent
FISH_SPEECH_URL = 'https://github.com/fishaudio/fish-speech.git'
FISH_SPEECH_REVISION = '1ff3c3ed2f80d3106a83909dd25b83129c3d816f'  # same as install.ps1
CHECKPOINT_REVISION = 'dd64a85cd82e9521f7202610ede97b367fc83946'  # same as install.ps1 / download.py


def run(*args, cwd=None, check=True):
    result = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise SystemExit(f'Failed: {" ".join(map(str, args))}\n{result.stdout}{result.stderr}')
    return result


def fish_speech(state):
    repo = state / 'fish-speech'
    if not (repo / '.git').is_dir():
        print('Fetching Fish Speech at', FISH_SPEECH_REVISION[:7])
        run('git', 'clone', '--filter=blob:none', '--no-checkout', FISH_SPEECH_URL, repo)
        run('git', '-C', repo, 'sparse-checkout', 'set', 'fish_speech', 'tools')
        run('git', '-C', repo, 'checkout', FISH_SPEECH_REVISION)
    head = run('git', '-C', repo, 'rev-parse', 'HEAD').stdout.strip()
    if head != FISH_SPEECH_REVISION:
        raise SystemExit(f'{repo} is at {head[:7]}, not the pinned {FISH_SPEECH_REVISION[:7]}; delete it and rerun.')
    patch = NODE / 'runtime' / 'streaming.patch'
    if run('git', '-C', repo, 'apply', '--reverse', '--check', patch, check=False).returncode:
        run('git', '-C', repo, 'apply', '--check', patch)
        run('git', '-C', repo, 'apply', patch)
        print('Applied the streaming patch')
    print('Fish Speech ready:', repo)


def model_location(state, checkpoint):
    location = state / 'model-location.json'
    if checkpoint is None:
        if not location.is_file():
            raise SystemExit('Pass --checkpoint-dir the first time.')
        checkpoint = Path(json.loads(location.read_text(encoding='utf-8-sig'))['checkpoint_dir'])
    revision = checkpoint / 'revision.txt'
    if revision.is_file() and revision.read_text().strip() != CHECKPOINT_REVISION:
        raise SystemExit(f'{checkpoint} holds another checkpoint revision; choose another folder.')
    location.write_text(json.dumps({'checkpoint_dir': str(checkpoint)}), encoding='utf-8')
    print('Checkpoint folder:', checkpoint)
    return checkpoint


def checkpoint_complete(checkpoint):
    index = checkpoint / 'model.safetensors.index.json'
    if not index.is_file() or not (checkpoint / 'revision.txt').is_file():
        return False
    shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
    return all((checkpoint / name).is_file() for name in shards | {'codec.pth', 'config.json', 'tokenizer.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint-dir', type=Path, help='Folder for the Drama 3 weights (about 41.5 GB).')
    parser.add_argument('--state', type=Path, default=NODE / 'local', help='State folder (default: this node\'s local/).')
    parser.add_argument('--download', action='store_true', help='Download the pinned checkpoint if it is incomplete.')
    parser.add_argument('--skip-check', action='store_true', help='Skip the import and FP8 kernel self-check.')
    args = parser.parse_args()
    state = args.state.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    print('Python:', sys.executable)
    fish_speech(state)
    checkpoint = model_location(state, args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None)
    if args.download and not checkpoint_complete(checkpoint):
        print('Downloading the pinned checkpoint (about 41.5 GB) ...')
        if subprocess.run([sys.executable, str(NODE / 'runtime' / 'download.py'), '--checkpoint-dir', str(checkpoint)]).returncode:
            raise SystemExit('Download failed; rerun to resume.')
    print('Checkpoint complete:', checkpoint_complete(checkpoint))
    if not args.skip_check:
        env = os.environ | {'DRAMA3_STATE': str(state)}
        check = subprocess.run([sys.executable, str(NODE / 'runtime' / 'selfcheck.py')], cwd=state, env=env,
                               capture_output=True, text=True)
        lines = [line for line in check.stdout.splitlines() if line.startswith('{')]
        print('Self-check:', lines[-1] if lines else (check.stdout + check.stderr)[-3000:])
        if check.returncode:
            raise SystemExit('Self-check failed.')
    print('Done. Restart ComfyUI to load the node.')


if __name__ == '__main__':
    main()
