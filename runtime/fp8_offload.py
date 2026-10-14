"""Experimental Drama 3 FP8 expert staging; upstream weights remain unchanged."""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY, EXPERT_KEY_PATTERN, STACKED_EXPERT_PARAMETER_PATTERN,
    FishOmniBackbone, backbone_name, load_backbone_weights, module_bytes,
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


def packed_layout(experts):
    """Where each of an expert's four tensors sits when they are stored together in one byte row.

    Returns ([(name, offset, size)], row size). Big weights first; every field starts 16-byte
    aligned, as the FP8 kernel's loads expect.
    """
    layout, offset = [], 0
    for name in ('w13_weight', 'w2_weight', 'w13_scale_inv', 'w2_scale_inv'):
        tensor = getattr(experts, name)
        size = tensor[0].numel() * tensor.element_size()
        layout.append((name, offset, size))
        offset += -(-size // 16) * 16
    return layout, offset


def packed_views(buffer, experts, layout):
    """The four expert tensors as views of a [rows, row size] uint8 buffer, one expert per row.

    The FP8 kernel takes strides, so it reads these views as it reads contiguous tensors.
    """
    views = {}
    for name, offset, size in layout:
        source = getattr(experts, name)
        views[name] = buffer[:, offset:offset + size].view(source.dtype).view(len(buffer), *source.shape[1:])
    return views


class ExpertCache:
    """Per-layer GPU cache of recently used experts, least recently used evicted first.

    Consecutive audio frames route mostly to the same experts (72 % repeat from the previous frame in
    a measured run), so most decode steps find their experts here and copy nothing from RAM. The FP8
    kernel reads the cache directly with remapped expert ids: same weights, same kernel.

    Each slot is one byte row holding all four of an expert's tensors (packed_layout), so an expert
    packed the same way in pinned RAM arrives in one copy instead of four.
    """
    def __init__(self, experts, capacity, device):
        self.capacity = capacity
        layout, size = packed_layout(experts)
        self.packed = torch.empty((capacity, size), dtype=torch.uint8, device=device)
        self.gpu = packed_views(self.packed, experts, layout)
        self.pending = None  # CUDA event of prefetch copies still in flight into this cache
        self.prefetched = set()  # experts prefetched since the last step of this layer
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
        self.experts_prefetched = 0
        self.prefetch_used = 0
        self.prefetch_wasted = 0
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
        slot_of = torch.tensor(self.place(experts, unique.tolist()), dtype=torch.long)
        remapped = slot_of[inverse].reshape_as(ids).to(self.device, non_blocking=True)
        return remapped, [cache.gpu[name] for name in FIELDS]

    def place(self, experts, wanted):
        """Put the distinct experts wanted in the layer's GPU cache; returns their slots, in order.

        Only missing experts are copied, least recently used ones not wanted now are evicted. Kept
        lean: fast mode calls this at every layer of every step while the GPU waits.
        """
        cache = experts.expert_cache
        if cache.pending is not None:  # prefetch copies may still be writing slots: let them land first
            torch.cuda.current_stream(cache.packed.device).wait_event(cache.pending)
            cache.pending = None
        slots, missing = cache.slots, []
        for e in wanted:
            if e in slots:
                slots.move_to_end(e)
            else:
                missing.append(e)
        if cache.prefetched:
            used = len(cache.prefetched.intersection(wanted))
            self.prefetch_used += used
            self.prefetch_wasted += len(cache.prefetched) - used
            cache.prefetched = set()
        if missing:
            needed = set(wanted)
            new_slots = []
            for e in missing:
                if cache.free:
                    slot = cache.free.pop()
                else:  # evict the least recently used expert this step does not need
                    slot = slots.pop(next(k for k in slots if k not in needed))
                slots[e] = slot
                new_slots.append(slot)
            self._fill(experts, cache, missing, new_slots)
        self.calls += 1
        self.cache_hits += len(wanted) - len(missing)
        self.cache_misses += len(missing)
        return [slots[e] for e in wanted]

    def prefetch(self, experts, candidates, limit, stream, after=None):
        """Copy up to limit of the candidate experts (best first) that the cache lacks, on stream.

        Fast mode guesses the next layer's experts and copies them while the GPU computes, instead of
        after the router has decided. The copies start after the event after (the current layer's own
        misses) and land in least recently used slots; place() waits for them before the layer's
        next use of the cache, so a wrong guess costs bandwidth and a slot, never correctness.
        """
        packed = getattr(experts, 'packed', None)
        cache = experts.expert_cache
        if packed is None:
            return
        picks = []
        for e in candidates:
            if e not in cache.slots and e not in picks:
                picks.append(e)
                if len(picks) == limit:
                    break
        if not picks:
            return
        if after is not None:
            stream.wait_event(after)
        slots = cache.slots
        with torch.cuda.stream(stream):
            for e in picks:
                slot = cache.free.pop() if cache.free else slots.pop(next(iter(slots)))
                slots[e] = slot
                cache.packed[slot].copy_(packed[e], non_blocking=True)
            cache.pending = torch.cuda.Event()
            cache.pending.record(stream)
        cache.prefetched.update(picks)
        self.experts_prefetched += len(picks)
        self.bytes_transferred += len(picks) * packed[0].numel()

    def _fill(self, experts, cache, missing, new_slots):
        """Copy experts missing into cache slots new_slots."""
        if getattr(experts, 'pinned', False):
            packed = getattr(experts, 'packed', None)
            for e, slot in zip(missing, new_slots):
                if packed is not None:  # same layout as the cache row: one copy
                    cache.packed[slot].copy_(packed[e], non_blocking=True)
                else:
                    for name in FIELDS:
                        cache.gpu[name][slot].copy_(getattr(experts, name)[e], non_blocking=True)
            self.bytes_transferred += len(missing) * sum(getattr(experts, n)[0].numel() * getattr(experts, n).element_size() for n in FIELDS)
            self.experts_transferred += len(missing)
        else:
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

    def stats(self):
        return {
            "offload_calls": self.calls,
            "experts_transferred": self.experts_transferred,
            "transferred_GiB": self.bytes_transferred / 2**30,
            "staging_capacity_experts": self.capacity,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "experts_prefetched": self.experts_prefetched,
            "prefetch_used": self.prefetch_used,
            "prefetch_wasted": self.prefetch_wasted,
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

    A pinned layer is packed: one buffer with one byte row per expert (packed_layout), and the four
    expert tensors become views of it, so the GPU cache receives each expert in one copy.
    """
    pinned = 0
    for layer in layers:
        experts = layer.mlp.experts
        if not isinstance(experts, OffloadedFP8Experts):
            continue
        if not getattr(experts, 'pinned', False):
            if not room_to_pin(experts, headroom_gib):
                break
            pack_experts(experts)
        pinned += 1
    return pinned


def room_to_pin(experts, headroom_gib=PIN_HEADROOM_GIB):
    """True when one more layer of experts fits in RAM with headroom to spare."""
    import psutil
    return psutil.virtual_memory().available - module_bytes(experts) >= headroom_gib * 2**30


def pack_experts(experts, copy=True):
    """Give a layer's experts one pinned buffer, one row per expert, with its four tensors as views.

    copy=False only allocates: the weights are then read straight into the rows (load_backbone_direct).
    """
    layout, row = packed_layout(experts)
    packed = torch.empty((experts.num_experts, row), dtype=torch.uint8, pin_memory=True)
    views = packed_views(packed, experts, layout)
    for name in FIELDS:
        if copy:
            views[name].copy_(getattr(experts, name))
        setattr(experts, name, views[name])  # a pageable original is freed here
    experts.packed = packed
    experts.pinned = True


SAFETENSORS_DTYPES = {'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16, 'F16': torch.float16,
                      'F32': torch.float32, 'I64': torch.int64, 'I32': torch.int32, 'U8': torch.uint8,
                      'BOOL': torch.bool}


def _shard_header(path):
    """(byte offset of the tensor data, {key: {dtype, shape, data_offsets}}) of a safetensors file."""
    with open(path, 'rb') as f:
        size = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(size))
    header.pop('__metadata__', None)
    return 8 + size, header


def _expected_slots(targets):
    """Every (parameter, expert, part) the checkpoint must fill, as Fish's load_backbone_weights counts them."""
    slots = set()
    for name, tensor in targets.items():
        match = STACKED_EXPERT_PARAMETER_PATTERN.fullmatch(name)
        if match is None:
            slots.add((name, -1, 'whole'))
            continue
        parts = ('gate', 'up') if match.group(2) == 'w13' else ('down',)
        slots.update((name, expert, part) for expert in range(tensor.shape[0]) for part in parts)
    return slots


@torch.inference_mode()
def load_backbone_direct(backbone, checkpoint_dir, threads=8):
    """Fish's load_backbone_weights for an FP8 checkpoint, with packed experts read straight from disk.

    The checkpoint stores each expert's gate, up and down projections (and their scales) as separate
    tensors: 61,440 of them. Every one lands in a contiguous byte range of its expert's packed row, so
    for packed layers the file bytes are read directly into pinned memory, one thread per shard: no
    per-tensor copies and no pinning copy afterwards. Other tensors are copied as Fish copies them.
    Every tensor is matched to its slot and checked (shape, and dtype for raw reads) before anything
    is read; a missing, duplicate or unknown tensor is refused.
    """
    targets = dict(backbone.named_parameters()) | dict(backbone.named_buffers())
    weight_map = json.loads((checkpoint_dir / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    plan, filled = {}, set()
    for shard in sorted(set(weight_map.values())):
        base, header = _shard_header(checkpoint_dir / shard)
        tasks = plan[shard] = []
        for key, info in header.items():
            if key.startswith(AUDIO_DECODER_PREFIXES):
                continue
            name = backbone_name(key)
            match = EXPERT_KEY_PATTERN.fullmatch(name)
            if match is None:
                if name not in targets:
                    raise ValueError(f'checkpoint key {key!r} has no parameter {name!r}')
                slot, destination, direct = (name, -1, 'whole'), targets[name], False
            else:
                prefix, expert = f"model.layers.{match['layer']}.mlp.experts", int(match['expert'])
                suffix = 'scale_inv' if match['kind'] == 'weight_scale_inv' else 'weight'
                if match['proj'] == 'down_proj':
                    slot_name, part = f'{prefix}.w2_{suffix}', 'down'
                    destination = targets[slot_name][expert]
                else:
                    slot_name, part = f'{prefix}.w13_{suffix}', 'gate' if match['proj'] == 'gate_proj' else 'up'
                    rows = targets[slot_name].shape[1] // 2
                    destination = targets[slot_name][expert, :rows] if part == 'gate' else targets[slot_name][expert, rows:]
                slot = (slot_name, expert, part)
                direct = getattr(backbone.get_submodule(prefix), 'packed', None) is not None
            dtype = SAFETENSORS_DTYPES.get(info['dtype'])
            if dtype is None:
                raise ValueError(f"{key}: unsupported checkpoint dtype {info['dtype']}")
            if tuple(info['shape']) != tuple(destination.shape):
                raise ValueError(f"{key}: checkpoint shape {info['shape']} != model shape {tuple(destination.shape)}")
            if (direct and dtype != destination.dtype) or (destination.dtype == torch.float8_e4m3fn and dtype != destination.dtype):
                raise ValueError(f"{key}: checkpoint dtype {info['dtype']} cannot fill {destination.dtype}")
            if slot in filled:
                raise ValueError(f'{slot} appears twice in the checkpoint')
            filled.add(slot)
            start, end = info['data_offsets']
            tasks.append((base + start, end - start, key, destination, direct, dtype))
    if missing := _expected_slots(targets) - filled:
        raise ValueError(f'{len(missing)} backbone tensors are not in the checkpoint, e.g. {sorted(missing)[:4]}')

    # Plain reads, no memory mapping: eight shards mapped at once beside 30 GB of pinned RAM made
    # Windows fail a load with "insufficient system resources" (error 1450).
    def read(shard, tasks):
        with torch.inference_mode(), open(checkpoint_dir / shard, 'rb', buffering=0) as raw:
            for offset, size, key, destination, direct, dtype in sorted(tasks, key=lambda task: task[0]):
                # packed experts land in their pinned rows; anything else in a CPU buffer, copied
                # (and converted, as Fish's loader does) into its parameter
                target = destination if direct else torch.empty(destination.shape, dtype=dtype)
                view = memoryview(target.reshape(-1).view(torch.uint8).numpy()).cast('B')
                if len(view) != size:
                    raise ValueError(f'{key}: {size} bytes in the file, {len(view)} expected')
                raw.seek(offset)
                done = 0
                while done < size:
                    count = raw.readinto(view[done:])
                    if not count:
                        raise ValueError(f'{shard} ends inside {key}')
                    done += count
                if not direct:
                    destination.copy_(target)

    with ThreadPoolExecutor(max(1, min(threads, len(plan)))) as pool:
        for future in [pool.submit(read, shard, tasks) for shard, tasks in plan.items()]:
            future.result()


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
        if index < resident:
            experts.to_empty(device=device)
        elif resident == 0 and room_to_pin(experts):
            pack_experts(experts, copy=False)  # load_backbone_direct reads the weights straight in
        else:
            experts.to_empty(device="cpu")
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
    if resident == 0:  # every layer streams: experts go straight from the files into pinned rows
        load_backbone_direct(backbone, checkpoint_dir)
    else:
        load_backbone_weights(backbone, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=AUDIO_DECODER_PREFIXES))
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    model = FishOmniModel(config, backbone, tower, tokenizer).eval()
    pinned = pin_offloaded_experts(layers)
    print(json.dumps({"pinned_expert_layers": pinned, "pageable_expert_layers": len(layers) - resident - pinned}), flush=True)
    model.offload_pool = pool
    print(f"Loaded in {time.perf_counter()-started:.1f}s", flush=True)
    return model
