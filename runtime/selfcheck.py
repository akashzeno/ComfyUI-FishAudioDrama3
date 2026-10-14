"""Check that Drama 3 can run in this Python before any weights are needed.

Importing run_drama3_full applies the same bootstrap a real job uses (state folder, Triton cache,
compiler, the natsort stand-in) and imports every Fish Speech module a job imports. Then Fish's own
FP8 MoE kernels run on random block-FP8 experts and are compared with the same maths in fp32.
Prints one JSON line; exit code 0 means pass.
"""
import json
import sys
import time

import torch

import run_drama3_full  # noqa: F401  (bootstrap + Fish Speech imports)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    FP8_BLOCK_SIZE, FP8_DTYPE, FP8_MAX, fp8_gemm_supported, fp8_moe_forward)


def quantize_blocks(weight):
    """[E, R, C] float -> FP8 weights and fp32 [E, R/128, C/128] scale_inv (dequant = q * s)."""
    e, r, c = weight.shape
    b = FP8_BLOCK_SIZE
    blocks = weight.reshape(e, r // b, b, c // b, b)
    scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-8) / FP8_MAX
    q = (blocks / scale[:, :, None, :, None]).to(FP8_DTYPE)
    return q.reshape(e, r, c), scale.float()


def dequantize(q, scale):
    e, r, c = q.shape
    b = FP8_BLOCK_SIZE
    return (q.float().reshape(e, r // b, b, c // b, b) * scale[:, :, None, :, None]).reshape(e, r, c)


def main():
    device = 'cuda:0'
    report = {'python': sys.executable, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'capability': '.'.join(map(str, torch.cuda.get_device_capability(0)))}
    report['fp8_supported'] = fp8_gemm_supported([device])
    if not report['fp8_supported']:
        print(json.dumps(report | {'ok': False, 'why': 'FP8 needs an SM89+ GPU and Triton'}))
        return 1
    import triton
    report['triton'] = triton.__version__
    torch.manual_seed(0)
    experts, hidden, inter, tokens, top_k = 8, 256, 256, 24, 4
    w13, s13 = quantize_blocks(torch.randn(experts, 2 * inter, hidden, device=device) * .05)
    w2, s2 = quantize_blocks(torch.randn(experts, hidden, inter, device=device) * .05)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]).to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1)
    started = time.perf_counter()
    out = fp8_moe_forward(x, ids, weights, w13, s13, w2, s2).float()
    torch.cuda.synchronize()
    report['kernel_seconds_incl_compile'] = round(time.perf_counter() - started, 2)
    g13, g2 = dequantize(w13, s13), dequantize(w2, s2)
    ref = torch.zeros(tokens, hidden, device=device)
    xf = x.float()
    for t in range(tokens):
        for slot in range(top_k):
            e = int(ids[t, slot])
            gate_up = g13[e] @ xf[t]
            h = torch.nn.functional.silu(gate_up[:inter]) * gate_up[inter:]
            ref[t] += weights[t, slot] * (g2[e] @ h)
    cosine = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    rel = ((out - ref).norm() / ref.norm()).item()
    report.update(cosine=round(cosine, 5), relative_error=round(rel, 4), finite=bool(torch.isfinite(out).all()))
    # Activations are FP8 too (W8A8), so a few percent of error is expected; a broken kernel is far off.
    report['ok'] = report['finite'] and cosine > .995 and rel < .08
    print(json.dumps(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
