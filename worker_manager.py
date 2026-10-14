"""Keep one Drama 3 job process alive across jobs (comfyenv mode).

Loading the codec and the 41.5 GB model takes about 1.5 minutes. A persistent worker
(`run_drama3_full.py --serve`) pays that once and then runs each queued job with the model
already in memory. It exits on its own after an idle period or when ComfyUI exits; the node also
stops it whenever ComfyUI is about to load another model onto the GPU.

No ComfyUI imports: the node passes in how to write to the terminal and how to check for an
interrupt, so this is testable with a fake worker.
"""
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

try:  # inside the ComfyUI package
    from .runtime import worker_protocol as wp
    from .log_echo import LogEcho
except ImportError:  # tests import it as a top-level module
    import worker_protocol as wp
    from log_echo import LogEcho

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class PersistentWorker:
    def __init__(self, temp_root, write):
        self.temp_root = Path(temp_root)
        self.write = write
        self.process = None
        self.dir = None
        self.key = None
        self.budget = None
        self.echo = None
        self._log = None

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def matches(self, key):
        return self.alive() and self.key == key

    def start(self, key, command, cwd, env, budget, idle_seconds):
        """Launch `command --serve <dir> ...`; call wait_ready() before submitting jobs."""
        self.stop('starting a new worker')
        self.dir = self.temp_root / f'drama3_worker_{uuid.uuid4().hex[:8]}'
        self.dir.mkdir(parents=True)
        log_path = self.dir / 'worker.log'
        self._log = log_path.open('w', encoding='utf-8')
        self.echo = LogEcho(log_path, self.write)
        self.key, self.budget = key, budget
        self.process = subprocess.Popen(
            [*map(str, command), '--serve', str(self.dir), '--gpu-budget', str(budget),
             '--idle-seconds', str(idle_seconds), '--parent-pid', str(os.getpid())],
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW, cwd=str(cwd), env=env)

    def wait_ready(self, check_interrupt, on_poll=None):
        while not wp.ready(self.dir):
            self._poll(check_interrupt, on_poll)
        self.echo.pump()

    def run(self, request, output, check_interrupt, on_poll=None):
        """Submit one job and wait for it. Raises with the worker's log if it fails or dies."""
        job_id = wp.submit(self.dir, {'request': str(request), 'output': str(output)})
        while (result := wp.result(self.dir, job_id)) is None:
            self._poll(check_interrupt, on_poll)
        if on_poll:
            on_poll()
        self.echo.pump()
        if not result['ok']:
            raise RuntimeError(f'Drama 3 generation failed:\n{(result.get("error") or "")[-6000:]}')
        return result

    def _poll(self, check_interrupt, on_poll):
        if self.process.poll() is not None:
            tail = self.log_tail()
            code = self.process.returncode
            self._cleanup()
            raise RuntimeError(f'The Drama 3 worker stopped unexpectedly (exit code {code}):\n{tail}')
        try:
            check_interrupt()
        except BaseException:
            self.kill()
            raise
        if on_poll:
            on_poll()
        self.echo.pump()
        time.sleep(.25)

    def log_tail(self, chars=6000):
        try:
            return (self.dir / 'worker.log').read_text(encoding='utf-8', errors='replace')[-chars:]
        except (OSError, TypeError):
            return ''

    def stop(self, reason=None, timeout=20):
        """Ask the worker to exit (it finishes nothing new); kill it if it does not."""
        if self.process is None:
            return
        if self.process.poll() is None:
            wp.request_stop(self.dir)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate()
            if reason:
                self.write(f'\n[Drama 3] worker stopped: {reason}\n')
        self._cleanup()

    def kill(self):
        if self.process is not None and self.process.poll() is None:
            self._terminate()
        self._cleanup()

    def _terminate(self):
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        else:
            self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _cleanup(self):
        if self.echo is not None:
            try:
                self.echo.pump()
            except OSError:
                pass
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.process = self.dir = self.key = self.budget = self.echo = None
