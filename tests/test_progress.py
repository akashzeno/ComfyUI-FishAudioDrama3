"""Progress reporting: run with DRAMA3_STATE set and runtime/ on PYTHONPATH (like test_speaker_dialogue)."""
import unittest

from run_drama3_full import FRAMES_PER_SECOND, estimate_frames, generation_progress


class EstimateFramesTests(unittest.TestCase):
    def test_spoken_text_sets_the_estimate(self):
        # Measured: a 111-character spoken line gave 160 frames.
        text = 'The weather report said rain by noon, so we packed two umbrellas, a flask of tea, and one very patient old dog.'
        self.assertAlmostEqual(estimate_frames(text), 160, delta=20)

    def test_directions_and_word_tags_are_not_spoken(self):
        plain = 'You mean when you swore it.'
        tagged = '[amused and teasing, laughing as she talks] You mean when you <emphasis>swore</emphasis> it.'
        self.assertEqual(estimate_frames(tagged), estimate_frames(plain))

    def test_sound_tags_add_time(self):
        self.assertGreater(estimate_frames('Fine. [laughs]'), estimate_frames('Fine.'))
        self.assertGreater(estimate_frames('Fine. [pause] Go.'), estimate_frames('Fine. Go.'))

    def test_frame_rate_matches_the_codec(self):
        self.assertAlmostEqual(FRAMES_PER_SECOND, 44100 / 2048)


class GenerationProgressTests(unittest.TestCase):
    def test_generation_spans_15_to_95_percent(self):
        self.assertAlmostEqual(generation_progress(0, 3, 0, 1, 0, 100), .15)
        self.assertAlmostEqual(generation_progress(2, 3, 0, 1, 100, 100), .95 * 1, delta=.01)

    def test_takes_advance_in_order(self):
        first = generation_progress(0, 3, 0, 1, 50, 100)
        second = generation_progress(1, 3, 0, 1, 50, 100)
        self.assertLess(first, second)

    def test_running_past_the_estimate_never_reaches_the_next_take(self):
        overrun = generation_progress(0, 3, 0, 1, 500, 100)
        self.assertLess(overrun, generation_progress(1, 3, 0, 1, 0, 100))


if __name__ == '__main__':
    unittest.main()
