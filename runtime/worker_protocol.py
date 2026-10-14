"""File-based job handoff between the ComfyUI node and a persistent Drama 3 worker.

server_dir/
  ready.json        written by the worker once the model and codec are loaded
  inbox/<id>.json   a job the node submitted (request path + output path)
  running/<id>.json the job the worker claimed
  done/<id>.json    {"ok": bool, "error": str | None}
  stop              the node asks the worker to exit

Every file is written to a temporary name and renamed into place, so neither side ever reads a
half-written file. Job ids sort in submission order. No torch or ComfyUI imports.
"""
import json
import os
import time
import uuid
from pathlib import Path


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(data), encoding='utf-8')
    os.replace(temp, path)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None


def submit(server_dir, payload):
    """Queue a job; returns its id."""
    job_id = f'{time.time_ns():020d}_{uuid.uuid4().hex[:8]}'
    _write(Path(server_dir) / 'inbox' / f'{job_id}.json', payload)
    return job_id


def claim_next(server_dir):
    """Worker side: take the oldest queued job, or None. A job is claimed exactly once."""
    inbox = Path(server_dir) / 'inbox'
    if not inbox.is_dir():
        return None
    for path in sorted(p for p in inbox.glob('*.json') if not p.name.startswith('.')):
        running = Path(server_dir) / 'running' / path.name
        running.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, running)
        except FileNotFoundError:
            continue
        return path.stem, _read(running)
    return None


def finish(server_dir, job_id, ok, error=None):
    _write(Path(server_dir) / 'done' / f'{job_id}.json', {'ok': ok, 'error': error})
    try:
        (Path(server_dir) / 'running' / f'{job_id}.json').unlink()
    except FileNotFoundError:
        pass


def result(server_dir, job_id):
    return _read(Path(server_dir) / 'done' / f'{job_id}.json')


def mark_ready(server_dir, info):
    _write(Path(server_dir) / 'ready.json', info)


def ready(server_dir):
    return _read(Path(server_dir) / 'ready.json')


def request_stop(server_dir):
    path = Path(server_dir) / 'stop'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('stop', encoding='utf-8')


def stop_requested(server_dir):
    return (Path(server_dir) / 'stop').exists()
