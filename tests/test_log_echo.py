import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from log_echo import LogEcho


class LogEchoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'generation.log'
        self.out = []
        self.echo = LogEcho(self.path, self.out.append)

    def tearDown(self):
        self.tmp.cleanup()

    def append(self, data):
        with self.path.open('ab') as file:
            file.write(data)

    def test_missing_log_is_not_an_error(self):
        self.echo.pump()
        self.assertEqual(self.out, [])

    def test_only_new_output_is_forwarded(self):
        self.append(b'loading\n')
        self.echo.pump()
        self.append(b'  3%|3  | 15/480\r  3%|3  | 16/480')
        self.echo.pump()
        self.assertEqual(''.join(self.out), 'loading\n  3%|3  | 15/480\r  3%|3  | 16/480')
        self.echo.pump()
        self.assertEqual(len(self.out), 2)

    def test_a_character_split_across_reads_survives(self):
        dash = '—'.encode('utf-8')
        self.append(b'umbrella ' + dash[:1])
        self.echo.pump()
        self.append(dash[1:] + b' bag')
        self.echo.pump()
        self.assertEqual(''.join(self.out), 'umbrella — bag')


if __name__ == '__main__':
    unittest.main()
