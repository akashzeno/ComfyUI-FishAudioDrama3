"""The expert cache must always hand the kernel the right experts, whatever it evicts.

Runs on the CPU with small fake experts. Needs the Fish Speech checkout like test_progress:
DRAMA3_STATE=<node>/local and PYTHONPATH=<node>/runtime.
"""
import random
import types
import unittest

import torch

import run_drama3_full  # noqa: F401  (puts Fish Speech on the path)
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fp8_offload import (FIELDS, ExpertCache, ExpertStagingPool, OffloadedFP8Experts, packed_layout,
                         packed_views, pin_offloaded_experts)


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

    def check_place(self, wanted):
        slots = self.pool.place(self.experts, wanted)
        cache = self.experts.expert_cache
        self.assertEqual(len(set(slots)), len(wanted), 'two experts share a slot')
        for name in FIELDS:
            for slot, expert in zip(slots, wanted):
                self.assertTrue(torch.equal(cache.gpu[name][slot], getattr(self.experts, name)[expert]),
                                f'{name}: slot {slot} is not expert {expert}')

    def test_place_returns_slots_holding_the_wanted_experts(self):
        self.check_place([3, 1, 4, 5])
        self.check_place([9, 3, 30, 1])

    def test_place_counts_hits_and_misses(self):
        self.check_place([3, 1, 4, 5])
        self.check_place([1, 4, 3, 9])
        self.assertEqual((self.pool.cache_hits, self.pool.cache_misses), (3, 5))

    def test_place_never_evicts_an_expert_the_step_needs(self):
        rng = random.Random(2)
        for _ in range(300):
            self.check_place(rng.sample(range(32), 8))  # 8 wanted, 8 slots

    def test_place_copies_pinned_experts_directly(self):
        self.experts.pinned = True  # the direct-copy path; the weights stay in ordinary CPU memory here
        rng = random.Random(3)
        for _ in range(100):
            self.check_place(rng.sample(range(32), 6))

    def test_place_copies_packed_experts_in_one_row(self):
        layout, size = packed_layout(self.experts)
        packed = torch.empty((32, size), dtype=torch.uint8)
        views = packed_views(packed, self.experts, layout)
        for name in FIELDS:
            views[name].copy_(getattr(self.experts, name))
            setattr(self.experts, name, views[name])
        self.experts.packed, self.experts.pinned = packed, True
        rng = random.Random(4)
        for _ in range(100):
            self.check_place(rng.sample(range(32), 6))

    def test_packed_views_hold_each_field_of_each_expert(self):
        layout, size = packed_layout(self.experts)
        packed = torch.zeros((32, size), dtype=torch.uint8)
        views = packed_views(packed, self.experts, layout)
        for name in FIELDS:
            views[name].copy_(getattr(self.experts, name))
        for name in FIELDS:
            self.assertTrue(torch.equal(views[name], getattr(self.experts, name)), name)
            self.assertEqual(views[name].untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())
        offsets = [offset for _, offset, _ in layout]
        self.assertTrue(all(offset % 16 == 0 for offset in offsets), 'every field must start 16-byte aligned')
        # one expert's row holds all four of its fields: copying the row copies the expert
        other = torch.zeros_like(packed)
        other[5].copy_(packed[7])
        for name in FIELDS:
            self.assertTrue(torch.equal(packed_views(other, self.experts, layout)[name][5], getattr(self.experts, name)[7]))

    @unittest.skipUnless(torch.cuda.is_available(), 'pinning needs CUDA')
    def test_pinning_packs_each_layer_into_one_buffer(self):
        experts = MoEExperts(8, 256, 128, fp8=True, device='cpu')
        for name in FIELDS:
            getattr(experts, name).copy_(torch.randn(getattr(experts, name).shape).to(getattr(experts, name).dtype))
        originals = {name: getattr(experts, name).clone() for name in FIELDS}
        experts.__class__ = OffloadedFP8Experts
        layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts))
        self.assertEqual(pin_offloaded_experts([layer], headroom_gib=0), 1)
        self.assertTrue(experts.packed.is_pinned())
        for name in FIELDS:
            field = getattr(experts, name)
            self.assertTrue(torch.equal(field.view(torch.uint8), originals[name].view(torch.uint8)), name)
            self.assertEqual(field.untyped_storage().data_ptr(), experts.packed.untyped_storage().data_ptr())

    def test_steps_larger_than_the_cache_use_the_staging_pool(self):
        self.check([list(range(12))])  # 12 unique > 8 slots
        self.assertEqual(self.pool.cache_misses, 0)


@unittest.skipUnless(torch.cuda.is_available(), 'prefetching copies on a CUDA stream')
class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.experts = FakeExperts()
        layout, size = packed_layout(self.experts)
        packed = torch.empty((32, size), dtype=torch.uint8, pin_memory=True)
        views = packed_views(packed, self.experts, layout)
        for name in FIELDS:
            views[name].copy_(getattr(self.experts, name))
            setattr(self.experts, name, views[name])
        self.experts.packed, self.experts.pinned = packed, True
        self.experts.expert_cache = ExpertCache(self.experts, capacity=8, device='cuda')
        self.pool = ExpertStagingPool('cuda')
        self.stream = torch.cuda.Stream()

    def assert_holds(self, expert):
        cache = self.experts.expert_cache
        for name in FIELDS:
            self.assertTrue(torch.equal(cache.gpu[name][cache.slots[expert]].cpu(), getattr(self.experts, name)[expert]),
                            f'{name}: expert {expert} not in its slot')

    def test_prefetch_copies_up_to_the_limit_of_candidates_not_cached(self):
        self.pool.place(self.experts, [1, 2, 3])
        self.pool.prefetch(self.experts, [2, 5, 1, 7, 9], limit=2, stream=self.stream)
        torch.cuda.synchronize()
        slots = self.experts.expert_cache.slots
        self.assertIn(5, slots)
        self.assertIn(7, slots)
        self.assertNotIn(9, slots)
        for expert in (1, 2, 3, 5, 7):
            self.assert_holds(expert)

    def test_prefetched_experts_are_hits_and_counted_used_or_wasted(self):
        self.pool.prefetch(self.experts, [5, 7], limit=2, stream=self.stream)
        self.pool.place(self.experts, [5, 1])
        torch.cuda.synchronize()
        self.assert_holds(5)
        self.assert_holds(1)
        self.assertEqual((self.pool.cache_hits, self.pool.cache_misses), (1, 1))
        self.assertEqual((self.pool.prefetch_used, self.pool.prefetch_wasted), (1, 1))

    def test_random_prefetch_and_place_keep_every_slot_right(self):
        rng = random.Random(5)
        for _ in range(200):
            self.pool.prefetch(self.experts, rng.sample(range(32), 6), limit=2, stream=self.stream)
            wanted = rng.sample(range(32), 5)
            slots = self.pool.place(self.experts, wanted)
            torch.cuda.synchronize()
            self.assertEqual(len(set(slots)), len(wanted))
            for expert in wanted:
                self.assert_holds(expert)


if __name__ == '__main__':
    unittest.main()
