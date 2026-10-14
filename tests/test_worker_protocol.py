import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

import worker_protocol as wp


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = Path(self.tmp.name) / 'worker'

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_to_claim_at_first(self):
        self.assertIsNone(wp.claim_next(self.server))
        self.assertIsNone(wp.result(self.server, 'missing'))
        self.assertIsNone(wp.ready(self.server))

    def test_jobs_are_claimed_once_and_in_order(self):
        first = wp.submit(self.server, {'request': 'a.json', 'output': 'a.wav'})
        second = wp.submit(self.server, {'request': 'b.json', 'output': 'b.wav'})
        job_id, payload = wp.claim_next(self.server)
        self.assertEqual((job_id, payload['request']), (first, 'a.json'))
        job_id, payload = wp.claim_next(self.server)
        self.assertEqual((job_id, payload['request']), (second, 'b.json'))
        self.assertIsNone(wp.claim_next(self.server))

    def test_result_round_trip(self):
        job_id = wp.submit(self.server, {'request': 'a.json', 'output': 'a.wav'})
        wp.claim_next(self.server)
        self.assertIsNone(wp.result(self.server, job_id))
        wp.finish(self.server, job_id, ok=False, error='boom')
        self.assertEqual(wp.result(self.server, job_id), {'ok': False, 'error': 'boom'})

    def test_ready_and_stop_flags(self):
        wp.mark_ready(self.server, {'pid': 7})
        self.assertEqual(wp.ready(self.server)['pid'], 7)
        self.assertFalse(wp.stop_requested(self.server))
        wp.request_stop(self.server)
        self.assertTrue(wp.stop_requested(self.server))


if __name__ == '__main__':
    unittest.main()
