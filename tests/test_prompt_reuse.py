"""A restored prompt state must continue exactly like a fresh prefill of that prompt.

Takes 2 and 3 of a job read the same prompt as take 1. PromptState keeps what prefill leaves in the
caches, so later takes restore it instead of reading the prompt again - but only if the restored
state is indistinguishable from a fresh prefill, whatever ran in between.

Needs an FP8-capable GPU (SM 8.9+) and the Fish Speech checkout like test_progress:
DRAMA3_STATE=<node>/local and PYTHONPATH=<node>/runtime.
"""
import unittest

import torch

from test_backbone_graphs import BUCKETS, tiny_backbone
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
from drama3_generate import BackboneGraphs, PromptState


def embeddings(rows, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn((1, rows, 256), generator=g, device='cuda').to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available() and fp8_gemm_supported(['cuda:0']), 'needs an FP8-capable GPU')
class PromptReuseTests(unittest.TestCase):
    def test_restored_state_continues_like_a_fresh_prefill(self):
        backbone = tiny_backbone((0, 2))
        with torch.inference_mode():
            graphs = BackboneGraphs.build(backbone, BUCKETS)
            prompt = embeddings(6, 1)
            hidden = backbone(inputs_embeds=prompt, input_pos=torch.arange(6, device='cuda'))[0, -1:].clone()
            saved = PromptState.capture(backbone, None, key=torch.arange(6), chunk=512, rows=6, hidden=hidden)
            expected = [graphs.step(embeddings(1, 10 + p), p).clone() for p in range(6, 12)]
            # another take decodes past the prompt, then a different prompt overwrites the caches
            for p in range(6, 15):
                graphs.step(embeddings(1, 50 + p), p)
            backbone(inputs_embeds=embeddings(9, 2), input_pos=torch.arange(9, device='cuda'))
            self.assertTrue(torch.equal(saved.restore(), hidden))
            got = [graphs.step(embeddings(1, 10 + p), p).clone() for p in range(6, 12)]
        for position, (a, b) in enumerate(zip(got, expected), start=6):
            self.assertTrue(torch.equal(a, b), f'position {position} differs after restoring the prompt state')

    def test_matches_only_the_same_prompt_and_chunking(self):
        backbone = tiny_backbone(())
        with torch.inference_mode():
            hidden = backbone(inputs_embeds=embeddings(4, 3), input_pos=torch.arange(4, device='cuda'))[0, -1:]
            saved = PromptState.capture(backbone, None, key=torch.tensor([[1, 2], [3, 4]]), chunk=512, rows=4,
                                        hidden=hidden)
        self.assertTrue(saved.matches(torch.tensor([[1, 2], [3, 4]]), 512))
        self.assertFalse(saved.matches(torch.tensor([[1, 2], [3, 5]]), 512))
        self.assertFalse(saved.matches(torch.tensor([[1, 2]]), 512))
        self.assertFalse(saved.matches(torch.tensor([[1, 2], [3, 4]]), 32))


if __name__ == '__main__':
    unittest.main()
