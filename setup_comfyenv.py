"""Set Drama 3 up to run with ComfyUI's own Python — no new environment, nothing pip-installed.

Run it with ComfyUI's Python (the comfyenv interpreter):

    python setup_comfyenv.py --checkpoint-dir D:\\Models\\drama-3            # code + self-check
    python setup_comfyenv.py --checkpoint-dir D:\\Models\\drama-3 --download # also the 41.5 GB weights
    python setup_comfyenv.py --checkpoint-dir D:\\Models\\drama-3 --verify   # weights you downloaded yourself

Steps, each skipped when already done:
  1. Fish Speech at the pinned commit (sparse checkout of fish_speech/ and tools/) into the state
     folder, with this repo's small streaming patch applied.
  2. model-location.json pointing at the checkpoint folder.
  3. --download: the pinned original FP8 checkpoint from fishaudio/drama-3 (runtime/download.py).
     --verify: for weights downloaded some other way (a download manager, say) — fetch the small
     files, check every file against Hugging Face's hashes, then mark the folder as complete.
  4. Self-check (runtime/selfcheck.py): every import a job needs, then Fish's FP8 MoE kernels on
     the GPU against the same maths in fp32 — run it before downloading the weights.

The pins are the ones install.ps1 uses, so both modes run the same code and the same checkpoint.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NODE = Path(__file__).resolve().parent
FISH_SPEECH_URL = 'https://github.com/fishaudio/fish-speech.git'
FISH_SPEECH_REVISION = '1ff3c3ed2f80d3106a83909dd25b83129c3d816f'  # same as install.ps1
CHECKPOINT_REVISION = 'dd64a85cd82e9521f7202610ede97b367fc83946'  # same as install.ps1 / download.py


# --- checking weights you downloaded yourself -------------------------------------------------
REPO = 'fishaudio/drama-3'


def needed(path):
    """Files the runtime reads (the same set runtime/download.py fetches)."""
    return path.endswith(('.safetensors', '.json')) or path in ('codec.pth', 'LICENSE', 'README.md')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def compare_files(folder, tree):
    """Check folder against a Hugging Face file tree: large files by SHA-256, small ones by git hash."""
    folder = Path(folder)
    files = [f for f in tree if f.get('type') == 'file' and needed(f['path'])]
    missing = [f['path'] for f in files if not (folder / f['path']).is_file()]
    present = [f for f in files if f['path'] not in missing]
    large = [f for f in present if f.get('lfs')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = dict(zip((f['path'] for f in large), pool.map(_sha256, (folder / f['path'] for f in large))))
    bad = []
    for f in present:
        path = folder / f['path']
        if f.get('lfs'):
            ok = path.stat().st_size == f['lfs']['size'] and digests[f['path']] == f['lfs']['oid']
        else:
            ok = _git_blob_sha1(path) == f['oid']
        if not ok:
            bad.append(f['path'])
    return {'missing': missing, 'bad': bad}


def verify_checkpoint(checkpoint):
    """Fetch the small files at the pinned revision, check every file, then write revision.txt."""
    base = f'https://huggingface.co/api/models/{REPO}/tree/{CHECKPOINT_REVISION}'
    tree = json.load(urllib.request.urlopen(base))
    checkpoint.mkdir(parents=True, exist_ok=True)
    for f in tree:
        if f.get('type') == 'file' and needed(f['path']) and not f.get('lfs') and not (checkpoint / f['path']).is_file():
            urllib.request.urlretrieve(f'https://huggingface.co/{REPO}/resolve/{CHECKPOINT_REVISION}/{f["path"]}',
                                       checkpoint / f['path'])
            print('Fetched', f['path'])
    print('Checking every file against Hugging Face\'s hashes ...')
    result = compare_files(checkpoint, tree)
    if result['missing'] or result['bad']:
        raise SystemExit(f'Not verified. Missing: {result["missing"] or "none"}; '
                         f'wrong contents (download again): {result["bad"] or "none"}')
    (checkpoint / 'revision.txt').write_text(CHECKPOINT_REVISION + '\n', encoding='utf-8')
    print('All files match; wrote revision.txt')


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
    parser.add_argument('--verify', action='store_true', help='Check weights you downloaded yourself against Hugging Face\'s hashes.')
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
    if args.verify:
        verify_checkpoint(checkpoint)
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
