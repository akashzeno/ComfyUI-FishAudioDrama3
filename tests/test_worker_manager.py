import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runtime'))

from worker_manager import PersistentWorker

FAKE = [sys.executable, '-u', str(ROOT / 'tests' / 'fake_worker.py')]


class Interrupted(Exception):
    pass


class WorkerManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lines = []
        self.worker = PersistentWorker(self.root, self.lines.append)
        self.never = lambda: None

    def tearDown(self):
        self.worker.stop('test over')
        self.tmp.cleanup()

    def request(self, text):
        job = self.root / f'job_{time.time_ns()}'
        job.mkdir()
        (job / 'request.json').write_text(json.dumps({'text': text}), encoding='utf-8')
        return job / 'request.json', job / 'speech.wav'

    def start(self, idle=60):
        self.worker.start(('fake',), FAKE, self.root, None, budget=9., idle_seconds=idle)
        self.worker.wait_ready(self.never)

    def test_runs_a_job_and_echoes_the_log(self):
        self.start()
        request, output = self.request('hello')
        self.worker.run(request, output, self.never)
        self.assertEqual(output.read_text(encoding='utf-8'), 'hello')
        self.assertIn('fake worker loading', ''.join(self.lines))

    def test_second_job_reuses_the_same_process(self):
        self.start()
        pid = self.worker.process.pid
        for text in ('one', 'two'):
            self.worker.run(*self.request(text), self.never)
        self.assertTrue(self.worker.matches(('fake',)))
        self.assertEqual(self.worker.process.pid, pid)
        self.assertFalse(self.worker.matches(('other',)))

    def test_failed_job_raises_and_keeps_the_worker(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            self.worker.run(*self.request('fail'), self.never)
        self.assertTrue(self.worker.alive())

    def test_dead_worker_is_reported(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'stopped unexpectedly'):
            self.worker.run(*self.request('die'), self.never)
        self.assertFalse(self.worker.alive())

    def test_stop_ends_the_process(self):
        self.start()
        process = self.worker.process
        self.worker.stop('done')
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.worker.alive())

    def test_interrupt_kills_the_worker_and_propagates(self):
        self.start()
        calls = []

        def interrupt():
            calls.append(1)
            if len(calls) > 2:
                raise Interrupted()
        with self.assertRaises(Interrupted):
            self.worker.run(*self.request('slow'), interrupt)
        self.assertFalse(self.worker.alive())

    def test_idle_worker_exits_by_itself(self):
        self.start(idle=.3)
        deadline = time.monotonic() + 10
        while self.worker.alive() and time.monotonic() < deadline:
            time.sleep(.1)
        self.assertFalse(self.worker.alive())


if __name__ == '__main__':
    unittest.main()
