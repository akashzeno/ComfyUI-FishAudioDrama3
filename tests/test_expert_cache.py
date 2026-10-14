"""The expert cache must always hand the kernel the right experts, whatever it evicts.

Runs on the CPU with small fake experts. Needs the Fish Speech checkout like test_progress:
DRAMA3_STATE=<node>/local and PYTHONPATH=<node>/runtime.
"""
import random
import unittest

import torch

import run_drama3_full  # noqa: F401  (puts Fish Speech on the path)
from fp8_offload import FIELDS, ExpertCache, ExpertStagingPool


class FakeExperts:
    num_experts = 32

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.w13_weight = torch.randn((32, 4, 3), generator=g)
        self.w13_scale_inv = torch.randn((32, 1, 1), generator=g)
        self.w2_weight = torch.randn((32, 3, 2), generator=g)
        self.w2_scale_inv = torch.randn((32, 1, 1), generator=g)


class ExpertCacheTests(unittest.TestCase):
    def setUp(self):
        self.experts = FakeExperts()
        self.experts.expert_cache = ExpertCache(self.experts, capacity=8, device='cpu')
        self.pool = ExpertStagingPool('cpu')

    def check(self, topk):
        ids = torch.tensor(topk, dtype=torch.long)
        remapped, tensors = self.pool.stage(self.experts, ids)
        for name, staged in zip(FIELDS, tensors):
            source = getattr(self.experts, name)
            for slot, expert in zip(remapped.flatten().tolist(), ids.flatten().tolist()):
                self.assertTrue(torch.equal(staged[slot], source[expert]), f'{name}: slot {slot} is not expert {expert}')

    def test_first_step_fills_the_cache(self):
        self.check([[3, 1, 4, 1]])
        self.assertEqual(self.pool.cache_misses, 3)

    def test_repeated_experts_are_hits(self):
        self.check([[3, 1, 4, 5]])
        self.check([[1, 4, 3, 9]])
        self.assertEqual((self.pool.cache_hits, self.pool.cache_misses), (3, 5))

    def test_eviction_never_drops_an_expert_the_step_needs(self):
        rng = random.Random(1)
        for _ in range(300):
            self.check([rng.sample(range(32), 8)])  # 8 wanted, 8 slots: every step evicts heavily

    def test_steps_larger_than_the_cache_use_the_staging_pool(self):
        self.check([list(range(12))])  # 12 unique > 8 slots
        self.assertEqual(self.pool.cache_misses, 0)


if __name__ == '__main__':
    unittest.main()
