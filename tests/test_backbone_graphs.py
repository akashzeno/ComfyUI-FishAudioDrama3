"""The graphed backbone step must equal Fish's eager graph-path step bit for bit.

A tiny Drama 3 backbone with real block-FP8 experts, some streamed from pinned RAM through the GPU
expert cache and some resident, runs decode steps both ways from the same state: eager
`backbone(..., static_kv_len=bucket)` (what Fish's own CUDA graphs capture) and
`BackboneGraphs.step`. Outputs and the attention caches and recurrent states left behind must match.

Needs an FP8-capable GPU (SM 8.9+) and the Fish Speech checkout like test_progress:
DRAMA3_STATE=<node>/local and PYTHONPATH=<node>/runtime.
"""
import unittest

import torch

import run_drama3_full  # noqa: F401  (puts Fish Speech on the path)
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
from drama3_generate import BackboneGraphs
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fp8_offload import (FIELDS, ExpertCache, ExpertStagingPool, OffloadedFP8Experts, packed_layout,
                         packed_views, pin_offloaded_experts)

BUCKETS = (8, 16, 32)


def tiny_config():
    return FishOmniTextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=4,
        layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, rope_theta=1e7,
        partial_rotary_factor=0.25, rms_norm_eps=1e-6, max_position_embeddings=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=128, shared_expert_intermediate_size=128)


def tiny_backbone(streamed, cache_slots=6):
    """Random weights; the layers in streamed keep their experts in pinned RAM behind a GPU cache."""
    torch.manual_seed(0)
    backbone = FishOmniBackbone(tiny_config(), fp8_experts=True, dtype=torch.bfloat16, device='cuda').eval()
    with torch.no_grad():
        for name, tensor in backbone.named_parameters():
            tensor.normal_(0, 0.02 if 'norm' not in name else 0.1)
        for layer in backbone.model.layers:
            experts = layer.mlp.experts
            for name in ('w13_weight', 'w2_weight'):
                getattr(experts, name).copy_(torch.randn(getattr(experts, name).shape, device='cuda').to(torch.float8_e4m3fn))
            for name in ('w13_scale_inv', 'w2_scale_inv'):
                getattr(experts, name).uniform_(0.01, 0.03)
    pool = ExpertStagingPool('cuda')
    for index in streamed:
        experts = backbone.model.layers[index].mlp.experts
        experts.to('cpu')
        experts.__class__ = OffloadedFP8Experts
        experts.staging_pool = pool
        if cache_slots:
            experts.expert_cache = ExpertCache(experts, cache_slots, 'cuda')
    pin_offloaded_experts(backbone.model.layers, headroom_gib=0)
    backbone.setup_caches(max_batch_size=1, max_seq_len=32)
    return backbone


def states(backbone):
    out = []
    for layer in backbone.model.layers:
        mixer = layer.self_attn if layer.layer_type == 'full_attention' else layer.linear_attn
        names = ('k_cache', 'v_cache') if layer.layer_type == 'full_attention' else ('conv_state', 'recurrent_state')
        out += [getattr(mixer, name) for name in names]
    return out


def bucket(position):
    return next(b for b in BUCKETS if b >= position + 1)


@unittest.skipUnless(torch.cuda.is_available() and fp8_gemm_supported(['cuda:0']), 'needs an FP8-capable GPU')
class BackboneGraphTests(unittest.TestCase):
    def check_layout(self, streamed):
        self.check_layout_on(tiny_backbone(streamed))

    def check_layout_on(self, backbone):
        with torch.inference_mode():
            graphs = BackboneGraphs.build(backbone, BUCKETS)
            self.assertIsNotNone(graphs)
            prompt = torch.randn((1, 5, 256), dtype=torch.bfloat16, device='cuda')
            backbone(inputs_embeds=prompt, input_pos=torch.arange(5, device='cuda'))
            for position in range(5, 20):  # crosses the 8 and 16 buckets
                embedding = torch.randn((1, 1, 256), dtype=torch.bfloat16, device='cuda')
                before = [s.clone() for s in states(backbone)]
                expected = backbone(inputs_embeds=embedding, input_pos=torch.tensor([position], device='cuda'),
                                    static_kv_len=bucket(position)).clone()
                expected_states = [s.clone() for s in states(backbone)]
                for state, saved in zip(states(backbone), before):
                    state.copy_(saved)
                got = graphs.step(embedding, position).clone()
                self.assertTrue(torch.isfinite(expected).all())
                self.assertTrue(torch.equal(got, expected), f'position {position}: output differs')
                for index, (state, wanted) in enumerate(zip(states(backbone), expected_states)):
                    self.assertTrue(torch.equal(state, wanted), f'position {position}: state {index} differs')

    def test_all_layers_streamed(self):
        self.check_layout((0, 1, 2, 3))

    def test_mixed_streamed_and_resident_layers(self):
        self.check_layout((1, 3))

    def test_all_layers_resident(self):
        self.check_layout(())

    def test_streamed_layers_prefetch_and_stay_exact(self):
        backbone = tiny_backbone((0, 1, 2, 3))
        pool = backbone.model.layers[0].mlp.experts.staging_pool
        self.check_layout_on(backbone)
        self.assertGreater(pool.experts_prefetched, 0, 'nothing was prefetched')
        self.assertEqual(pool.prefetch_used + pool.prefetch_wasted, pool.experts_prefetched)

    def test_streamed_layers_without_a_cache_are_not_graphed(self):
        backbone = tiny_backbone((1, 3), cache_slots=0)
        with torch.inference_mode():
            self.assertIsNone(BackboneGraphs.build(backbone, BUCKETS))

    def test_streamed_experts_still_count_hits_and_misses(self):
        backbone = tiny_backbone((0, 1, 2, 3))
        pool = backbone.model.layers[0].mlp.experts.staging_pool
        with torch.inference_mode():
            graphs = BackboneGraphs.build(backbone, BUCKETS)
            backbone(inputs_embeds=torch.randn((1, 5, 256), dtype=torch.bfloat16, device='cuda'),
                     input_pos=torch.arange(5, device='cuda'))
            before = pool.cache_hits + pool.cache_misses
            graphs.step(torch.randn((1, 1, 256), dtype=torch.bfloat16, device='cuda'), 5)
        self.assertEqual(pool.cache_hits + pool.cache_misses - before, 4 * 4)  # 4 layers x top 4



@unittest.skipUnless(torch.cuda.is_available() and fp8_gemm_supported(['cuda:0']), 'needs an FP8-capable GPU')
class PackedExpertTests(unittest.TestCase):
    def test_kernel_reads_packed_experts_exactly_like_contiguous_ones(self):
        torch.manual_seed(0)
        experts = MoEExperts(16, 256, 128, fp8=True, device='cuda')
        for name in ('w13_weight', 'w2_weight'):
            getattr(experts, name).copy_(torch.randn(getattr(experts, name).shape, device='cuda').to(torch.float8_e4m3fn))
        for name in ('w13_scale_inv', 'w2_scale_inv'):
            getattr(experts, name).uniform_(0.01, 0.03)
        layout, size = packed_layout(experts)
        packed = torch.empty((16, size), dtype=torch.uint8, device='cuda')
        views = packed_views(packed, experts, layout)
        for name in FIELDS:
            views[name].copy_(getattr(experts, name))
        for tokens in (1, 5):
            x = torch.randn((tokens, 256), device='cuda').to(torch.bfloat16)
            ids = torch.stack([torch.randperm(16, device='cuda')[:4] for _ in range(tokens)])
            weights = torch.rand((tokens, 4), device='cuda').to(torch.bfloat16)
            contiguous = fp8_moe_forward(x, ids, weights, *(getattr(experts, name) for name in FIELDS))
            from_packed = fp8_moe_forward(x, ids, weights, *(views[name] for name in FIELDS))
            self.assertTrue(torch.equal(contiguous, from_packed), f'{tokens} token(s): packed experts differ')


if __name__ == '__main__':
    unittest.main()
