"""Experimental Drama 3 FP8 expert staging; upstream weights remain unchanged."""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY,
    FishOmniBackbone, load_backbone_weights, module_bytes,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    fp8_gemm_supported, fp8_moe_forward,
)
from fish_speech.models.text2semantic.fish_omni.model import (
    AUDIO_DECODER_PREFIXES, FishOmniModel, iter_checkpoint_tensors,
)
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fish_speech.models.text2semantic.fish_omni.tower import (
    FishOmniAudioDecoder, load_audio_decoder_weights,
)

FIELDS = ("w13_weight", "w13_scale_inv", "w2_weight", "w2_scale_inv")


def gather_rows(source, ids, out):
    """out[i] = source[ids[i]]. One index_select: it already copies on torch's own threads, and
    splitting it across more threads measured 3-5x slower (they contend for memory bandwidth)."""
    torch.index_select(source, 0, ids, out=out)


def _rows(tensor):
    return tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor


class ExpertCache:
    """Per-layer GPU cache of recently used experts, least recently used evicted first.

    Consecutive audio frames route mostly to the same experts (72 % repeat from the previous frame in
    a measured run), so most decode steps find their experts here and copy nothing from RAM. The FP8
    kernel reads the cache directly with remapped expert ids: same weights, same kernel.
    """
    def __init__(self, experts, capacity, device):
        self.capacity = capacity
        self.gpu = {name: torch.empty((capacity, *getattr(experts, name).shape[1:]), dtype=getattr(experts, name).dtype,
                                      device=device) for name in FIELDS}
        self.slots = OrderedDict()  # expert id -> slot, oldest first
        self.free = list(range(capacity - 1, -1, -1))

class ExpertStagingPool:
    """One reusable pinned-host/GPU pool, shared by sequential offloaded layers.

    The top-k GPU-to-CPU read waits for preceding work on the current stream.
    That also protects pinned source memory from reuse before its copy completes.
    It reads the router on the CPU, so it runs between CUDA graphs, never inside one
    (drama3_generate.BackboneGraphs calls it between pieces); concurrent requests must not share it.
    """
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.host = {}
        self.gpu = {}
        self.capacity = 0
        self.calls = 0
        self.bytes_transferred = 0
        self.experts_transferred = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.shape_signature = None

    def stage(self, experts, topk_ids):
        ids = topk_ids.detach().cpu()
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        count = unique.numel()
        cache = getattr(experts, 'expert_cache', None)
        if cache is not None and count <= cache.capacity:
            return self._stage_cached(experts, cache, ids, unique, inverse)
        self._ensure(experts, count)
        staged = []
        if getattr(experts, 'pinned', False):
            self._copy_direct(experts, unique.tolist(), self.gpu)
        for name in FIELDS:
            source = getattr(experts, name)
            host = self.host[name][:count]
            gpu = self.gpu[name][:count]
            if not getattr(experts, 'pinned', False):
                gather_rows(_rows(source), unique, _rows(host))
                gpu.copy_(host, non_blocking=True)
            self.bytes_transferred += gpu.numel() * gpu.element_size()
            staged.append(gpu)
        self.calls += 1
        self.experts_transferred += count
        remapped = inverse.reshape_as(ids).to(self.device, non_blocking=True)
        return remapped, staged

    def _copy_direct(self, experts, ids, targets):
        """Copy experts[ids[i]] into targets[name][i] straight from pinned RAM (asynchronous DMA)."""
        for i, expert in enumerate(ids):
            for name in FIELDS:
                targets[name][i].copy_(getattr(experts, name)[expert], non_blocking=True)

    def _ensure(self, experts, count):
        signature = tuple((tuple(getattr(experts, k).shape[1:]), getattr(experts, k).dtype) for k in FIELDS)
        if self.shape_signature not in (None, signature):
            raise ValueError("Staging pool cannot mix expert tensor shapes")
        self.shape_signature = signature
        if count > self.capacity:
            self.capacity = min(experts.num_experts, 2 ** math.ceil(math.log2(max(8, count))))
            self.host.clear()
            self.gpu.clear()
            for name in FIELDS:
                source = getattr(experts, name)
                shape = (self.capacity, *source.shape[1:])
                self.host[name] = torch.empty(shape, dtype=source.dtype, pin_memory=True)
                self.gpu[name] = torch.empty(shape, dtype=source.dtype, device=self.device)

    def _stage_cached(self, experts, cache, ids, unique, inverse):
        wanted = unique.tolist()
        missing = [e for e in wanted if e not in cache.slots]
        for e in wanted:
            if e in cache.slots:
                cache.slots.move_to_end(e)
        needed = set(wanted)
        new_slots = []
        for e in missing:
            if cache.free:
                slot = cache.free.pop()
            else:  # evict the least recently used expert this step does not need
                victim = next(k for k in cache.slots if k not in needed)
                slot = cache.slots.pop(victim)
            cache.slots[e] = slot
            new_slots.append(slot)
        if missing and getattr(experts, 'pinned', False):
            for e, slot in zip(missing, new_slots):
                for name in FIELDS:
                    cache.gpu[name][slot].copy_(getattr(experts, name)[e], non_blocking=True)
            self.bytes_transferred += len(missing) * sum(getattr(experts, n)[0].numel() * getattr(experts, n).element_size() for n in FIELDS)
            self.experts_transferred += len(missing)
        elif missing:
            count = len(missing)
            self._ensure(experts, count)
            miss = torch.tensor(missing, dtype=torch.long)
            slots = torch.tensor(new_slots, dtype=torch.long).to(self.device, non_blocking=True)
            for name in FIELDS:
                host = self.host[name][:count]
                gpu = self.gpu[name][:count]
                gather_rows(_rows(getattr(experts, name)), miss, _rows(host))
                gpu.copy_(host, non_blocking=True)
                _rows(cache.gpu[name]).index_copy_(0, slots, _rows(gpu))
                self.bytes_transferred += gpu.numel() * gpu.element_size()
            self.experts_transferred += count
        self.calls += 1
        self.cache_hits += len(wanted) - len(missing)
        self.cache_misses += len(missing)
        slot_of = torch.tensor([cache.slots[e] for e in wanted], dtype=torch.long)
        remapped = slot_of[inverse].reshape_as(ids).to(self.device, non_blocking=True)
        return remapped, [cache.gpu[name] for name in FIELDS]

    def stats(self):
        return {
            "offload_calls": self.calls,
            "experts_transferred": self.experts_transferred,
            "transferred_GiB": self.bytes_transferred / 2**30,
            "staging_capacity_experts": self.capacity,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


class OffloadedFP8Experts(MoEExperts):
    def forward(self, x, topk_ids, topk_weights):
        if not self.fp8 or not x.is_cuda or self.w13_weight.device.type != "cpu":
            raise RuntimeError("Offloaded experts require FP8 CPU weights and CUDA activations")
        remapped, tensors = self.staging_pool.stage(self, topk_ids)
        return fp8_moe_forward(x, remapped, topk_weights, *tensors)


PIN_HEADROOM_GIB = 1.0  # spare RAM needed beyond the layer being swapped


def pin_offloaded_experts(layers, headroom_gib=PIN_HEADROOM_GIB):
    """Move streamed experts into pinned (page-locked) RAM.

    Pageable expert memory gets trimmed from the working set when RAM runs low, and every copy then
    stalls on page faults (measured: about 1 GB/s instead of 10). Pinned memory stays resident, and
    the GPU copies each expert straight from it at full PCIe speed with no CPU copy in between.

    The experts are already in RAM, so pinning swaps a pageable copy for a pinned one: the total does
    not grow, but each layer briefly exists twice. A layer is pinned only when that transient copy
    fits with headroom to spare; the rest keep the pageable path. Returns how many were pinned.
    """
    import psutil
    pinned = 0
    for layer in layers:
        experts = layer.mlp.experts
        if not isinstance(experts, OffloadedFP8Experts):
            continue
        size = module_bytes(experts)
        if psutil.virtual_memory().available - size < headroom_gib * 2**30:
            break
        for name in FIELDS:
            tensor = getattr(experts, name)
            tensor.data = tensor.data.pin_memory()
        experts.pinned = True
        pinned += 1
    return pinned


@torch.inference_mode()
def load_offloaded_model(checkpoint_dir: str | Path, gpu_budget_gib=26.0, device="cuda:0"):
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    required = set(index["weight_map"].values()) | {"config.json", "tokenizer.json"}
    missing = sorted(name for name in required if not (checkpoint_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Checkpoint download is incomplete: {len(missing)} files missing ({', '.join(missing[:3])}).")
    if not fp8_gemm_supported([device]):
        raise RuntimeError("FP8 requires working Triton and an SM89+ NVIDIA GPU")
    config = FishOmniConfig.from_dict(json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8")))
    if not config.fp8_experts:
        raise ValueError("This experiment expects the original FP8 expert checkpoint")
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.bfloat16)
        tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    layers = backbone.model.layers
    expert_bytes = [module_bytes(layer.mlp.experts) for layer in layers]
    dense_bytes = module_bytes(backbone) + module_bytes(tower) - sum(expert_bytes)
    pool_bytes = max(expert_bytes)  # worst-case prefill selects all experts of one layer
    budget = int(gpu_budget_gib * 2**30)
    if dense_bytes + pool_bytes > budget:
        raise ValueError("GPU budget is too small for dense weights and expert staging")
    used = dense_bytes + pool_bytes
    per_expert = expert_bytes[0] // layers[0].mlp.experts.num_experts
    if used + sum(expert_bytes) <= budget:
        resident, cache_capacity = len(layers), 0  # everything fits: nothing to stream
    else:
        # Stream every layer and spend the remaining VRAM on per-layer expert caches: a cache slot
        # serves every layer, a resident layer only one (measured: 16 slots/layer hit 87 %).
        resident = 0
        cache_capacity = min(64, int((budget - used) // (per_expert * len(layers))))
        if cache_capacity < 8:  # one decode step needs 8 experts per layer
            cache_capacity = 0
        used += cache_capacity * per_expert * len(layers)
    pool = ExpertStagingPool(device)
    backbone.model.embed_tokens.to_empty(device=device)
    backbone.model.norm.to_empty(device=device)
    backbone.lm_head.to_empty(device=device)
    for index, layer in enumerate(layers):
        for name, child in layer.named_children():
            if name != "mlp":
                child.to_empty(device=device)
        for name, child in layer.mlp.named_children():
            if name != "experts":
                child.to_empty(device=device)
        experts = layer.mlp.experts
        experts.to_empty(device=device if index < resident else "cpu")
        if index >= resident:
            experts.__class__ = OffloadedFP8Experts
            experts.staging_pool = pool
            if cache_capacity:
                experts.expert_cache = ExpertCache(experts, cache_capacity, device)
    backbone.inverse_frequencies.clear()
    tower.to_empty(device=device)
    print(json.dumps({
        "resident_expert_layers": resident,
        "offloaded_expert_layers": len(layers) - resident,
        "dense_weights_GiB": dense_bytes / 2**30,
        "resident_weights_GiB": (used-pool_bytes) / 2**30,
        "cpu_expert_weights_GiB": sum(expert_bytes[resident:]) / 2**30,
        "maximum_staging_GiB": pool_bytes / 2**30,
        "expert_cache_slots_per_layer": cache_capacity,
        "expert_cache_GiB": cache_capacity * per_expert * len(layers) / 2**30,
    }), flush=True)
    started = time.perf_counter()
    load_backbone_weights(backbone, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=AUDIO_DECODER_PREFIXES))
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    model = FishOmniModel(config, backbone, tower, tokenizer).eval()
    pinned = pin_offloaded_experts(layers)
    print(json.dumps({"pinned_expert_layers": pinned, "pageable_expert_layers": len(layers) - resident - pinned}), flush=True)
    model.offload_pool = pool
    print(f"Loaded in {time.perf_counter()-started:.1f}s", flush=True)
    return model
