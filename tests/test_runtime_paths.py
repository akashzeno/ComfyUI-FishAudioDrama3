import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_paths import resolve_launch


class ResolveLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.node = self.root / 'node'
        (self.node / 'runtime').mkdir(parents=True)
        (self.node / 'runtime' / 'run_drama3_full.py').write_text('')
        self.python = str(self.root / 'comfyenv' / 'python.exe')

    def tearDown(self):
        self.tmp.cleanup()

    def isolated_runtime(self):
        runtime = self.root / 'Drama3Runtime'
        (runtime / 'drama3-env' / 'Scripts').mkdir(parents=True)
        (runtime / 'drama3-env' / 'Scripts' / 'python.exe').write_text('')
        (runtime / 'run_drama3_full.py').write_text('')
        return runtime

    def test_defaults_to_comfyui_python_and_node_state_folder(self):
        launch = resolve_launch(self.node, '', env={}, executable=self.python)
        self.assertEqual(launch['mode'], 'comfyenv')
        self.assertEqual(launch['python'], Path(self.python))
        self.assertEqual(launch['runner'], self.node / 'runtime' / 'run_drama3_full.py')
        self.assertEqual(launch['cwd'], self.node / 'local')
        self.assertEqual(launch['env']['DRAMA3_STATE'], str(self.node / 'local'))

    def test_runtime_folder_with_isolated_env_keeps_upstream_behaviour(self):
        runtime = self.isolated_runtime()
        launch = resolve_launch(self.node, str(runtime), env={}, executable=self.python)
        self.assertEqual(launch['mode'], 'isolated')
        self.assertEqual(launch['python'], runtime / 'drama3-env' / 'Scripts' / 'python.exe')
        self.assertEqual(launch['runner'], runtime / 'run_drama3_full.py')
        self.assertEqual(launch['cwd'], runtime)
        self.assertNotIn('DRAMA3_STATE', launch['env'])

    def test_env_variable_selects_isolated_runtime(self):
        runtime = self.isolated_runtime()
        launch = resolve_launch(self.node, '', env={'DRAMA3_RUNTIME': str(runtime)}, executable=self.python)
        self.assertEqual(launch['mode'], 'isolated')

    def test_runtime_folder_without_isolated_env_is_a_state_folder(self):
        state = self.root / 'state'
        launch = resolve_launch(self.node, str(state), env={}, executable=self.python)
        self.assertEqual(launch['mode'], 'comfyenv')
        self.assertEqual(launch['cwd'], state)
        self.assertEqual(launch['env']['DRAMA3_STATE'], str(state))

    def test_missing_runner_is_reported(self):
        os.remove(self.node / 'runtime' / 'run_drama3_full.py')
        with self.assertRaises(RuntimeError):
            resolve_launch(self.node, '', env={}, executable=self.python)


if __name__ == '__main__':
    unittest.main()
