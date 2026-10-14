"""Faster loading must load exactly the same model.

Needs the Fish Speech checkout like test_progress: DRAMA3_STATE=<node>/local and
PYTHONPATH=<node>/runtime.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from run_drama3_full import CODEC_MASK_SIZE, unused_codec_masks_skipped
from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX, STACKED_EXPERT_PARAMETER_PATTERN, FishOmniBackbone, load_backbone_weights)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, iter_checkpoint_tensors
from fp8_offload import load_backbone_direct, pack_experts


def tiny_config():
    return FishOmniTextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=4,
        layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, rope_theta=1e7,
        partial_rotary_factor=0.25, rms_norm_eps=1e-6, max_position_embeddings=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=128, shared_expert_intermediate_size=128)


def blank_backbone():
    return FishOmniBackbone(tiny_config(), fp8_experts=True, dtype=torch.bfloat16, device='cpu').eval()


def tensors_of(backbone):
    return dict(backbone.named_parameters()) | dict(backbone.named_buffers())


def write_checkpoint(folder, extra=None, drop=None, shards=2):
    """A random tiny backbone saved the way the Drama 3 checkpoint stores it: one tensor per expert projection."""
    torch.manual_seed(0)
    source = blank_backbone()
    with torch.no_grad():
        for name, tensor in tensors_of(source).items():
            if tensor.dtype == torch.float8_e4m3fn:
                tensor.copy_(torch.randn(tensor.shape).to(torch.float8_e4m3fn))
            else:
                tensor.copy_(torch.randn(tensor.shape).to(tensor.dtype))
    tensors = {'audio_decoder.layers.0.weight': torch.randn(3, 3)}  # skipped: the audio decoder loads separately
    for name, tensor in tensors_of(source).items():
        match = STACKED_EXPERT_PARAMETER_PATTERN.fullmatch(name)
        if match is None:
            key = 'lm_head.weight' if name == 'lm_head.weight' else CHECKPOINT_BACKBONE_PREFIX + name[len('model.'):]
            tensors[key] = tensor.detach().clone()
            continue
        base = CHECKPOINT_BACKBONE_PREFIX + match.group(1)[len('model.'):]
        kind = 'weight' if match.group(3) == 'weight' else 'weight_scale_inv'
        for expert in range(tensor.shape[0]):
            if match.group(2) == 'w2':
                tensors[f'{base}.{expert}.down_proj.{kind}'] = tensor[expert].clone()
            else:
                half = tensor.shape[1] // 2
                tensors[f'{base}.{expert}.gate_proj.{kind}'] = tensor[expert, :half].clone()
                tensors[f'{base}.{expert}.up_proj.{kind}'] = tensor[expert, half:].clone()
    tensors.update(extra or {})
    for key in drop or ():
        del tensors[key]
    keys = sorted(tensors)
    weight_map = {}
    for shard in range(shards):
        name = f'model-{shard + 1:05d}-of-{shards:05d}.safetensors'
        part = {key: tensors[key].contiguous() for key in keys[shard::shards]}
        save_file(part, str(Path(folder) / name))
        weight_map.update({key: name for key in part})
    (Path(folder) / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}), encoding='utf-8')
    return source


def same_bytes(a, b):
    return torch.equal(a.detach().cpu().contiguous().view(torch.uint8), b.detach().cpu().contiguous().view(torch.uint8))


class CodecMaskTests(unittest.TestCase):
    def test_the_unused_codec_masks_are_not_built(self):
        real_tril = torch.tril
        with unused_codec_masks_skipped():
            mask = torch.tril(torch.ones(CODEC_MASK_SIZE, CODEC_MASK_SIZE, dtype=torch.bool))
            small = torch.tril(torch.ones(5, 5, dtype=torch.bool))
            other = torch.tril(torch.ones(CODEC_MASK_SIZE, 4))
        self.assertEqual(mask.numel(), 0, 'the 32768 x 32768 mask was built')
        self.assertTrue(torch.equal(small, real_tril(torch.ones(5, 5, dtype=torch.bool))))
        self.assertTrue(torch.equal(other, real_tril(torch.ones(CODEC_MASK_SIZE, 4))))
        self.assertIs(torch.tril, real_tril, 'torch.tril was not restored')

    def test_tril_is_restored_after_an_error(self):
        real_tril = torch.tril
        with self.assertRaises(RuntimeError):
            with unused_codec_masks_skipped():
                raise RuntimeError('codec failed to load')
        self.assertIs(torch.tril, real_tril)


@unittest.skipUnless(torch.cuda.is_available(), 'pinned memory needs CUDA')
class DirectLoadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def packed_backbone(self, packed_layers):
        backbone = blank_backbone()
        for index in packed_layers:  # the rest stay ordinary CPU tensors
            pack_experts(backbone.model.layers[index].mlp.experts, copy=False)
        return backbone

    def test_direct_load_matches_fish_loader_byte_for_byte(self):
        source = write_checkpoint(self.folder.name)
        reference = blank_backbone()
        load_backbone_weights(reference, iter_checkpoint_tensors(Path(self.folder.name), skip_prefixes=AUDIO_DECODER_PREFIXES))
        direct = self.packed_backbone((0, 2, 3))
        load_backbone_direct(direct, Path(self.folder.name))
        expected, got, original = tensors_of(reference), tensors_of(direct), tensors_of(source)
        self.assertEqual(set(expected), set(got))
        for name in expected:
            self.assertTrue(same_bytes(expected[name], original[name]), f'{name}: Fish loader differs from the source')
            self.assertTrue(same_bytes(got[name], expected[name]), f'{name}: direct load differs')
        experts = direct.model.layers[2].mlp.experts
        self.assertEqual(experts.w13_weight.untyped_storage().data_ptr(), experts.packed.untyped_storage().data_ptr())

    def test_a_missing_expert_tensor_is_refused_before_reading(self):
        write_checkpoint(self.folder.name, drop=['model.language_model.layers.1.mlp.experts.7.up_proj.weight'])
        with self.assertRaisesRegex(ValueError, 'not in the checkpoint'):
            load_backbone_direct(self.packed_backbone((0, 1, 2, 3)), Path(self.folder.name))

    def test_a_wrongly_shaped_tensor_is_refused(self):
        write_checkpoint(self.folder.name, extra={
            'model.language_model.layers.0.mlp.experts.3.down_proj.weight': torch.zeros(4, 4, dtype=torch.float8_e4m3fn)})
        with self.assertRaisesRegex(ValueError, 'shape'):
            load_backbone_direct(self.packed_backbone((0, 1, 2, 3)), Path(self.folder.name))

    def test_an_unknown_tensor_is_refused(self):
        write_checkpoint(self.folder.name, extra={'model.language_model.layers.0.mystery.weight': torch.zeros(2)})
        with self.assertRaisesRegex(ValueError, 'no parameter'):
            load_backbone_direct(self.packed_backbone(()), Path(self.folder.name))


if __name__ == '__main__':
    unittest.main()
