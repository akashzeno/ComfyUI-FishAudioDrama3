import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setup_comfyenv import compare_files, needed


def blob_sha1(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


class CompareFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        big, small = b'weights' * 100, b'{"a": 1}'
        (self.folder / 'model-00001-of-00001.safetensors').write_bytes(big)
        (self.folder / 'config.json').write_bytes(small)
        self.tree = [
            {'type': 'file', 'path': 'model-00001-of-00001.safetensors', 'size': len(big), 'oid': 'x',
             'lfs': {'oid': hashlib.sha256(big).hexdigest(), 'size': len(big)}},
            {'type': 'file', 'path': 'config.json', 'size': len(small), 'oid': blob_sha1(small)},
            {'type': 'file', 'path': 'chat_template.jinja', 'size': 3, 'oid': 'y'},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_files_the_runtime_needs_are_checked(self):
        self.assertEqual([f['path'] for f in self.tree if needed(f['path'])],
                         ['model-00001-of-00001.safetensors', 'config.json'])

    def test_matching_files_pass(self):
        self.assertEqual(compare_files(self.folder, self.tree), {'missing': [], 'bad': []})

    def test_a_corrupted_large_file_is_caught(self):
        (self.folder / 'model-00001-of-00001.safetensors').write_bytes(b'weightz' * 100)
        self.assertEqual(compare_files(self.folder, self.tree)['bad'], ['model-00001-of-00001.safetensors'])

    def test_a_changed_small_file_is_caught(self):
        (self.folder / 'config.json').write_bytes(b'{"a": 2}')
        self.assertEqual(compare_files(self.folder, self.tree)['bad'], ['config.json'])

    def test_missing_files_are_listed(self):
        (self.folder / 'config.json').unlink()
        self.assertEqual(compare_files(self.folder, self.tree)['missing'], ['config.json'])


if __name__ == '__main__':
    unittest.main()
