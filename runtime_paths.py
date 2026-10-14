"""Decide which Python runs a Drama 3 job, and from where.

comfyenv mode (this fork's default): the job runs in a separate process started with ComfyUI's
own interpreter, so nothing is installed anywhere. The runner stays in this repo's ``runtime/``
folder; the pinned Fish Speech checkout, model-location.json and caches live in a state folder
(``local/`` by default, passed to the runner as ``DRAMA3_STATE``).

isolated mode (upstream behaviour): a runtime folder made by install.ps1, recognised by its
``drama3-env`` virtual environment, runs with that environment's Python.

No ComfyUI imports here, so the logic is testable on its own.
"""
from pathlib import Path

RUNNER = 'run_drama3_full.py'
ISOLATED_PYTHON = Path('drama3-env') / 'Scripts' / 'python.exe'


def resolve_launch(node_dir, runtime_folder='', env=None, executable=None, config_folder=''):
    """Return {'mode', 'python', 'runner', 'cwd', 'env'} for one job.

    Precedence for the folder: the node's runtime_folder input, then DRAMA3_RUNTIME, then
    runtime.json (config_folder). A folder holding an isolated env selects isolated mode; any
    other folder is the comfyenv state folder. With none given, the state folder is node/local.
    """
    node_dir = Path(node_dir)
    env = env or {}
    chosen = (runtime_folder or '').strip() or env.get('DRAMA3_RUNTIME', '').strip() or (config_folder or '').strip()
    folder = Path(chosen).expanduser().resolve() if chosen else None
    if folder is not None and (folder / ISOLATED_PYTHON).is_file():
        runner = folder / RUNNER
        if not runner.is_file():
            raise RuntimeError(f'The isolated Drama 3 runtime in {folder} has no {RUNNER}.')
        return {'mode': 'isolated', 'python': folder / ISOLATED_PYTHON, 'runner': runner, 'cwd': folder, 'env': {}}
    runner = node_dir / 'runtime' / RUNNER
    if not runner.is_file():
        raise RuntimeError(f'{runner} is missing; reinstall the node.')
    state = folder if folder is not None else node_dir / 'local'
    return {'mode': 'comfyenv', 'python': Path(executable), 'runner': runner, 'cwd': state,
            'env': {'DRAMA3_STATE': str(state)}}
