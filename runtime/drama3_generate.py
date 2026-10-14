"""Drama 3 decoding replayed from CUDA graphs, with the routed experts streamed from RAM.

Eager decoding is bound by kernel launches: thousands of small kernels per audio frame. Fish Speech
captures a whole decode step (lm_head -> audio decoder -> backbone) as CUDA graphs, but with experts
streamed from RAM one graph cannot span a step: every streamed layer stops on the CPU to fetch the
experts its router picked. So the step is captured in pieces:

- the audio decoder (the "tower"): lm_head -> sample a frame -> the next step's input embedding;
- the backbone, cut at each streamed layer's router: the CPU reads the router's picks, copies the
  missing experts into that layer's GPU cache and writes their slots to a static buffer, then the
  next piece (those experts, then the next layer's attention and router) is replayed.

The pieces run Fish's own graph code path - attention over padded cache buckets, sampling settings
held on the GPU - so takes repeat exactly but are not bit-identical to the fully eager path.
Graphs are captured once per loaded model, before any seed is set: capturing runs the sampler and
would otherwise shift the random stream of the first take.
"""
import os
import time

import torch
from loguru import logger
from tqdm import tqdm

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_GRAPH_CAPACITY, DRAMA3_GRAPH_KV_BUCKETS, ensure_drama3_caches)
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext
from fish_speech.models.text2semantic.fish_omni.tower import RAS_WINDOW_FRAMES

from fp8_offload import FIELDS, OffloadedFP8Experts


# Fast mode copies up to 2 experts per layer before the router has picked them: the next layer's
# router, applied early to this layer's state, finds 53% of the real picks. Replaying 1,947 recorded
# frames, 2 per layer modelled 14% faster decoding; 3 or more lost time, the wrong guesses crowding
# the real copies off PCIe.
PREFETCH_PER_LAYER = 2
PREFETCH_CANDIDATES = 16


def capture_graph(run, pool, stream):
    """Warm run() up on a side stream, then capture it into a CUDA graph.

    Share one warm-up stream between captures: the allocator keeps each stream's freed blocks for
    that stream alone, so a new stream per capture leaves every warm-up's memory stranded
    (measured: 0.86 GiB over the backbone's pieces).
    """
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        run()
    return graph


def bucket_for(buckets, rows):
    return next(b for b in buckets if b >= rows)


class TowerGraphs:
    """CUDA graphs of lm_head -> audio decoder (sample a frame) -> next input embedding."""

    def __init__(self, model, buckets, pool, stream):
        backbone, tower = model.backbone, model.audio_decoder
        self.model = model
        device = tower.device
        width = 1 + model.config.audio_decoder.num_codebooks
        self.hidden = torch.zeros((1, model.config.text.hidden_size), dtype=backbone.dtype, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.frame = torch.zeros((1, width), dtype=torch.long, device=device)
        self.sampled = torch.zeros_like(self.frame)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=device)
        self.temperature = torch.zeros((), dtype=torch.float32, device=device)
        self.top_k = torch.zeros((), dtype=torch.long, device=device)
        self.top_p = torch.zeros((), dtype=torch.float32, device=device)
        self.finite = torch.ones((), dtype=torch.bool, device=device)
        self.embeddings = None
        self.buckets = buckets
        self.graphs = {kv_len: self.capture(kv_len, pool, stream) for kv_len in buckets}

    def tower_attentions(self):
        tower = self.model.audio_decoder
        for layer in (*tower.shared_layers, *tower.residual_layers):
            yield layer.transformer_layer.attention

    def run(self):
        backbone, tower = self.model.backbone, self.model.audio_decoder
        logits = backbone.compute_logits(self.hidden)
        self.finite.copy_(torch.isfinite(logits).all())
        sampled = tower.decode_step(logits, self.hidden, self.position, self.frame, self.history,
                                    temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                                    check_finite=False)
        self.sampled.copy_(sampled)
        embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
        if self.embeddings is None:  # first warm-up run, outside capture
            self.embeddings = torch.empty_like(embeddings[None])
        self.embeddings.copy_(embeddings[None])

    def capture(self, kv_len, pool, stream):
        for attention in self.tower_attentions():
            attention.static_kv_len = kv_len
        try:
            return capture_graph(self.run, pool, stream)
        finally:
            for attention in self.tower_attentions():
                attention.static_kv_len = None

    def start(self, hidden, position, prompt_frames, temperature, top_k, top_p):
        """Load the state after prefill: last hidden row, its position and frame."""
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.frame.copy_(prompt_frames[-1:])
        window = prompt_frames[-RAS_WINDOW_FRAMES:, 0]
        self.history.fill_(-1)
        self.history[RAS_WINDOW_FRAMES - len(window):].copy_(window)
        self.temperature.fill_(temperature)
        self.top_k.fill_(top_k)
        self.top_p.fill_(top_p)

    def step(self, position):
        """Sample the frame after position; returns (frame, main token, next input embedding [1, 1, H])."""
        self.graphs[bucket_for(self.buckets, position + 1)].replay()
        sampled = self.sampled.clone()
        main, finite = torch.stack((sampled[0, 0], self.finite.long())).tolist()
        if not finite:
            raise FloatingPointError('non-finite logits; refusing to sample invalid audio')
        return sampled, main, self.embeddings

    def advance(self, hidden, sampled):
        """After the backbone step: the next step's hidden row, frame, history and position."""
        self.hidden.copy_(hidden)
        self.frame.copy_(sampled)
        self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
        self.position.add_(1)


class BackboneGraphs:
    """CUDA graphs of one backbone decode step, cut at every streamed layer's router.

    Piece s runs the experts of the previous streamed layer, the resident layers after it, and the
    attention and router of the next streamed layer. Between pieces the CPU fetches that layer's
    experts into its GPU cache. With every layer streamed, a step is 41 graph launches instead of
    about 6,800 kernel launches.

    Each piece also runs the following layer's router early, on the state it has; its best guesses
    not in the cache are copied on a side stream while the next piece computes (prefetch).
    """

    def __init__(self, backbone, buckets, pool, stream):
        self.backbone = backbone
        self.layers = list(backbone.model.layers)
        self.buckets = buckets
        self.stops = [i for i, layer in enumerate(self.layers) if isinstance(layer.mlp.experts, OffloadedFP8Experts)]
        device = backbone.model.norm.weight.device
        hidden = backbone.config.hidden_size
        top_k = self.layers[0].mlp.top_k
        rotary = backbone.config.rotary_dim
        frequencies = backbone.inverse_frequencies.get(device)
        if frequencies is None:  # as FishOmniBackbone.run_layers computes them
            exponents = torch.arange(0, rotary, 2, dtype=torch.float32, device=device)
            frequencies = 1.0 / (backbone.config.rope_theta ** (exponents / rotary))
        self.frequencies = frequencies
        self.embeddings = torch.zeros((1, 1, hidden), dtype=backbone.dtype, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.cos = torch.zeros((1, rotary), dtype=torch.float32, device=device)
        self.sin = torch.zeros_like(self.cos)
        self.residual = torch.zeros((1, 1, hidden), dtype=backbone.residual_dtype, device=device)
        self.routed = torch.zeros((1, hidden), dtype=backbone.dtype, device=device)
        self.weights = torch.zeros((1, top_k), dtype=backbone.dtype, device=device)
        # Layers whose next layer is prefetched: streamed, and packed for one-copy experts.
        self.guess = {i: self.layers[i + 1] for i in self.stops
                      if i + 1 in self.stops and getattr(self.layers[i + 1].mlp.experts, 'packed', None) is not None}
        self.top_k = top_k
        # The router's picks, then the early guesses for the next layer: one copy back per stop.
        self.picks = torch.zeros((1, top_k + PREFETCH_CANDIDATES), dtype=torch.long, device=device)
        self.ids = self.picks[:, :top_k]
        self.slots = torch.zeros((1, top_k), dtype=torch.long, device=device)
        self.picks_host = torch.zeros(self.picks.shape, dtype=torch.long, pin_memory=True)
        self.slots_host = torch.zeros((1, top_k), dtype=torch.long, pin_memory=True)
        self.picks_array, self.slots_array = self.picks_host.numpy(), self.slots_host.numpy()
        self.prefetch_stream = torch.cuda.Stream(device)
        self.misses_queued = torch.cuda.Event()
        self.output = torch.zeros((1, 1, hidden), dtype=backbone.dtype, device=device)
        self.graphs = {}
        for piece in range(len(self.stops) + 1):
            for kv_len in (buckets if self.attends(piece) else (None,)):
                self.graphs[piece, kv_len] = capture_graph(
                    lambda: self.run_piece(piece, kv_len or buckets[-1]), pool, stream)

    @classmethod
    def build(cls, backbone, buckets, pool=None, stream=None):
        """Capture the graphs, or None when a layer cannot be graphed (several devices, no expert cache)."""
        device = backbone.model.norm.weight.device
        if device.type != 'cuda':
            return None
        for layer in backbone.model.layers:
            if layer.input_layernorm.weight.device != device:
                return None
            experts = layer.mlp.experts
            if isinstance(experts, OffloadedFP8Experts):
                cache = getattr(experts, 'expert_cache', None)
                if cache is None or cache.capacity < layer.mlp.top_k:
                    return None
            elif experts.w13_weight.device != device:
                return None
        with torch.cuda.device(device):
            return cls(backbone, tuple(buckets), pool if pool is not None else torch.cuda.graph_pool_handle(),
                       stream if stream is not None else torch.cuda.Stream())

    def piece_layers(self, piece):
        """The layers whose attention runs in this piece."""
        first = self.stops[piece - 1] + 1 if piece else 0
        last = self.stops[piece] + 1 if piece < len(self.stops) else len(self.layers)
        return range(first, last)

    def attends(self, piece):
        return any(self.layers[i].layer_type == 'full_attention' for i in self.piece_layers(piece))

    def run_piece(self, piece, kv_len):
        context = LayerContext(input_pos=self.position, cos=self.cos, sin=self.sin, kv_len=kv_len,
                               continue_state=True, use_cache=True)
        if piece == 0:
            angles = self.position.float()[:, None] * self.frequencies[None, :]
            angles = torch.cat((angles, angles), dim=-1)
            self.cos.copy_(angles.cos())
            self.sin.copy_(angles.sin())
            x = self.embeddings.to(self.backbone.residual_dtype)
        else:
            x = self.finish_experts(self.layers[self.stops[piece - 1]])
        for index in self.piece_layers(piece):
            layer = self.layers[index]
            if piece < len(self.stops) and index == self.stops[piece]:
                self.attend_and_route(index, x, context)
                return
            x = layer(x, context)
        norm = self.backbone.model.norm
        self.output.copy_(norm(x).to(self.backbone.dtype))

    def attend_and_route(self, index, x, context):
        """DecoderLayer.forward up to the router (SparseMoeBlock.forward's first half)."""
        layer = self.layers[index]
        compute_dtype = layer.input_layernorm.weight.dtype
        mixer = layer.self_attn if layer.layer_type == 'full_attention' else layer.linear_attn
        x = x + mixer(layer.input_layernorm(x).to(compute_dtype), context)
        flat = layer.post_attention_layernorm(x).to(compute_dtype).reshape(-1, x.shape[-1])
        moe = layer.mlp
        probabilities = torch.softmax(moe.gate(flat).float(), dim=-1)
        weights, ids = probabilities.topk(moe.top_k, dim=-1)
        if moe.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        self.residual.copy_(x)
        self.routed.copy_(flat)
        self.weights.copy_(weights.to(flat.dtype))
        self.ids.copy_(ids)
        following = self.guess.get(index)
        if following is not None:  # the next layer's router, early, on this layer's state
            early = following.post_attention_layernorm(x).to(compute_dtype).reshape(-1, x.shape[-1])
            self.picks[:, self.top_k:].copy_(following.mlp.gate(early).topk(PREFETCH_CANDIDATES, dim=-1).indices)

    def finish_experts(self, layer):
        """The rest of the layer: the fetched experts from the cache, the shared expert, the residual."""
        moe = layer.mlp
        cache = moe.experts.expert_cache
        out = fp8_moe_forward(self.routed, self.slots, self.weights, *(cache.gpu[name] for name in FIELDS))
        shared_gate = torch.sigmoid(moe.shared_expert_gate(self.routed))
        return self.residual + (out + shared_gate * moe.shared_expert(self.routed)).view_as(self.residual)

    def fetch(self, index):
        """Read the router's picks, copy missing experts into the cache, point the kernel at their slots.

        Returns the next layer's guessed experts, best first, when that layer is prefetched.
        """
        experts = self.layers[index].mlp.experts
        self.picks_host.copy_(self.picks)  # waits for the piece that ran this layer's router
        picks = self.picks_array[0].tolist()
        # top-k picks are distinct. The host buffers are rewritten only after the next piece has
        # run, and with it the slot copy queued here.
        self.slots_array[0] = experts.staging_pool.place(experts, picks[:self.top_k])
        self.slots.copy_(self.slots_host, non_blocking=True)
        self.misses_queued.record()
        return picks[self.top_k:] if index in self.guess else None

    def step(self, embeddings, position):
        """The backbone at position for input embeddings [1, 1, H]; returns the final-norm hidden [1, 1, H].

        The result is a static buffer, overwritten by the next step.
        """
        kv_len = bucket_for(self.buckets, position + 1)
        self.embeddings.copy_(embeddings)
        self.position.fill_(position)
        for piece in range(len(self.stops) + 1):
            guesses = self.fetch(self.stops[piece - 1]) if piece else None
            self.graphs[piece, kv_len if self.attends(piece) else None].replay()
            if guesses:  # copy the next layer's likely experts while this piece computes
                following = self.guess[self.stops[piece - 1]].mlp.experts
                following.staging_pool.prefetch(following, guesses, PREFETCH_PER_LAYER, self.prefetch_stream,
                                                after=self.misses_queued)
        return self.output


class PromptState:
    """What reading a prompt leaves in the caches, kept so the next take of that prompt can skip it.

    The takes of a job read the same prompt. Prefill writes the prompt's rows of the backbone's and
    the audio decoder's attention caches and the linear-attention layers' running state; decoding
    only writes past the prompt. Restoring these equals reading the prompt again, whatever ran in
    between (tests/test_prompt_reuse.py), and saves the prefill: about 2 s per take.
    """

    def __init__(self, backbone, tower, key, chunk, rows, hidden, copies):
        self.backbone, self.tower, self.key, self.chunk, self.rows = backbone, tower, key, chunk, rows
        self.hidden, self.copies = hidden, copies

    @staticmethod
    def targets(backbone, tower, rows):
        """The cache regions prefill writes for a prompt of rows rows."""
        for layer in backbone.model.layers:
            if layer.layer_type == 'full_attention':
                yield layer.self_attn.k_cache[:, :, :rows]
                yield layer.self_attn.v_cache[:, :, :rows]
            else:
                yield layer.linear_attn.conv_state
                yield layer.linear_attn.recurrent_state
        if tower is not None:  # row p of the tower is labelled with frame p + 1: the last row is decode's
            for layer in (*tower.shared_layers, *tower.residual_layers):
                yield layer.transformer_layer.attention.k_cache[:, :, :rows - 1]
                yield layer.transformer_layer.attention.v_cache[:, :, :rows - 1]

    @classmethod
    def capture(cls, backbone, tower, *, key, chunk, rows, hidden):
        """Copy the state right after prefilling the prompt key (in chunks of chunk rows)."""
        copies = [target.clone() for target in cls.targets(backbone, tower, rows)]
        return cls(backbone, tower, key.clone(), chunk, rows, hidden.clone(), copies)

    def matches(self, key, chunk):
        return chunk == self.chunk and key.shape == self.key.shape and torch.equal(key, self.key)

    def restore(self):
        """Put the prompt's state back; returns the prompt's last hidden row."""
        for target, copy in zip(self.targets(self.backbone, self.tower, self.rows), self.copies):
            target.copy_(copy)
        return self.hidden


class DecodeGraphs:
    """The audio decoder's and, when possible, the backbone's decode-step graphs for one model."""

    def __init__(self, model, tower, backbone, capacity):
        self.model, self.tower, self.backbone, self.capacity = model, tower, backbone, capacity
        self.cache_rows = model.cache_capacity
        self.prompt_state = None  # the last prompt read, for the next take of the same prompt

    @classmethod
    @torch.inference_mode()
    def build(cls, model):
        """Capture the graphs for a loaded model, or None when they cannot or should not be used."""
        if os.environ.get('DRAMA3_FAST_MODE', '1') == '0' or model.audio_decoder.device.type != 'cuda':
            return None
        capacity = min(DRAMA3_GRAPH_CAPACITY, model.config.text.max_position_embeddings)
        buckets = tuple(sorted({min(b, capacity) for b in DRAMA3_GRAPH_KV_BUCKETS}))
        ensure_drama3_caches(model, capacity)
        pool = torch.cuda.graph_pool_handle()
        device = model.audio_decoder.device
        started, before = time.perf_counter(), torch.cuda.memory_reserved(device)
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
            tower = TowerGraphs(model, buckets, pool, stream)
        after_tower = torch.cuda.memory_reserved(device)
        backbone = BackboneGraphs.build(model.backbone, buckets, pool, stream)
        logger.info(f'Drama 3 CUDA graphs: audio decoder{" and backbone" if backbone else ""}, kv buckets '
                    f'{buckets}, captured in {time.perf_counter() - started:.1f}s, VRAM reserved '
                    f'+{(after_tower - before) / 2**30:.2f} GiB (decoder) '
                    f'+{(torch.cuda.memory_reserved(device) - after_tower) / 2**30:.2f} GiB (backbone), '
                    f'allocated {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB')
        return cls(model, tower, backbone, capacity)

    def fits(self, rows):
        """True when a request of rows prompt + new frames can use these graphs.

        The graphs captured the attention caches; eager generation of a longer request reallocates
        them, after which the graphs must not be replayed.
        """
        return rows <= self.capacity and self.model.cache_capacity == self.cache_rows


@torch.inference_mode()
def generate_drama3_graphed(model, prompt, *, graphs, max_new_tokens, temperature, top_k, top_p,
                            prefill_chunk_size, frame_callback=None, stream_interval=64):
    """Fish's generate_drama3 with the decode step replayed from CUDA graphs.

    Prefill is unchanged, and skipped when the previous take read the same prompt (PromptState).
    Returns CPU [frames, 1 + N] without EOS, and the finish reason.
    """
    if top_k < 1:
        raise InvalidInputError(f'top_k must be at least 1; got {top_k}')
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or not graphs.fits(capacity):
        raise ValueError(f'max_new_tokens={max_new_tokens} does not fit after {prompt_rows} prompt rows')
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=tower.device)
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)

    started = time.perf_counter()
    state = graphs.prompt_state
    if state is not None and state.matches(prompt, prefill_chunk_size):
        last = state.restore()
        logger.info(f'Drama 3 prefill: {prompt_rows} rows restored from the previous take')
    else:
        graphs.prompt_state = None  # free the old copies before reading the new prompt
        for start in range(0, prompt_rows, prefill_chunk_size):
            end = min(start + prefill_chunk_size, prompt_rows)
            chunk = frames[start:end]
            embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
            hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])
            hidden = hidden[0].to(tower.device)
            paired_end = min(end, prompt_rows - 1)
            if paired_end > start:
                tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                              frames[start:paired_end], frames[start + 1: paired_end + 1])
        last = hidden[-1:]
        graphs.prompt_state = PromptState.capture(backbone, tower, key=prompt, chunk=prefill_chunk_size,
                                                  rows=prompt_rows, hidden=last)
        logger.info(f'Drama 3 prefill: {prompt_rows} rows in {time.perf_counter() - started:.2f}s')

    graphs.tower.start(last, prompt_rows - 1, frames[:prompt_rows], temperature, top_k, top_p)
    produced, finish_reason = 0, 'length'
    for step in tqdm(range(max_new_tokens)):
        position = prompt_rows - 1 + step
        sampled, main, embeddings = graphs.tower.step(position)
        if main == model.config.eos_token_id:
            finish_reason = 'eos'
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if frame_callback is not None and produced % stream_interval == 0:
            frame_callback(frames[prompt_rows: prompt_rows + produced].cpu())
        if step + 1 < max_new_tokens:
            if graphs.backbone is not None:
                hidden = graphs.backbone.step(embeddings, position + 1)
            else:
                hidden = backbone(inputs_embeds=embeddings, input_pos=positions[position + 1: position + 2])
            graphs.tower.advance(hidden[0].to(tower.device), sampled)
    result = frames[prompt_rows: prompt_rows + produced].cpu()
    if frame_callback is not None and produced % stream_interval:
        frame_callback(result)
    return result, finish_reason
