"""Drama 3 generation with the audio decoder captured as CUDA graphs, the backbone run eagerly.

Fish Speech captures a whole decode step (lm_head -> audio decoder -> backbone) as CUDA graphs,
because eager decoding is bound by kernel launches. With experts streamed from RAM the backbone
cannot be captured: every layer stops on the CPU to fetch the experts its router picked. The audio
decoder (the "tower") has no such stop, so this captures lm_head + tower + the next step's input
embedding and leaves only the backbone eager. The captured part is Fish's own graph code path —
padded attention buckets and device-side sampling parameters — so takes are Fish's default numerics,
not bit-identical to the fully eager path.

Graphs are captured once per loaded model, before any seed is set: capturing runs the sampler and
would otherwise shift the random stream of the first take.
"""
import os
import time

import torch
from loguru import logger
from tqdm import tqdm

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_GRAPH_CAPACITY, DRAMA3_GRAPH_KV_BUCKETS, ensure_drama3_caches)
from fish_speech.models.text2semantic.fish_omni.tower import RAS_WINDOW_FRAMES


class TowerDecodeGraphs:
    """CUDA graphs of lm_head -> audio decoder (sample a frame) -> next input embedding."""

    def __init__(self, model):
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
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capacity = min(DRAMA3_GRAPH_CAPACITY, model.config.text.max_position_embeddings)
        self.buckets = tuple(sorted({min(b, self.capacity) for b in DRAMA3_GRAPH_KV_BUCKETS}))

    @classmethod
    @torch.inference_mode()
    def build(cls, model):
        """Capture the graphs for a loaded model, or None when they cannot or should not be used."""
        if os.environ.get('DRAMA3_CUDA_GRAPHS', '1') == '0' or model.audio_decoder.device.type != 'cuda':
            return None
        ensure_drama3_caches(model, min(DRAMA3_GRAPH_CAPACITY, model.config.text.max_position_embeddings))
        graphs = cls(model)
        started = time.perf_counter()
        with torch.cuda.device(model.audio_decoder.device):
            for kv_len in graphs.buckets:
                graphs.capture(kv_len)
        logger.info(f'Drama 3 audio-decoder CUDA graphs: kv buckets {graphs.buckets} captured in '
                    f'{time.perf_counter() - started:.1f}s')
        return graphs

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
            self.embeddings = torch.empty_like(embeddings)
        self.embeddings.copy_(embeddings)

    def capture(self, kv_len):
        for attention in self.tower_attentions():
            attention.static_kv_len = kv_len
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self.run()
            self.graphs[kv_len] = graph
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
        """Sample the frame after position; returns (frame, main token, next input embedding)."""
        kv_len = next(b for b in self.buckets if b >= position + 1)
        self.graphs[kv_len].replay()
        sampled = self.sampled.clone()
        main, finite = torch.stack((sampled[0, 0], self.finite.long())).tolist()
        if not finite:
            raise FloatingPointError('non-finite logits; refusing to sample invalid audio')
        return sampled, main, self.embeddings

    def advance(self, hidden, sampled):
        """After the eager backbone step: the next step's hidden row, frame, history and position."""
        self.hidden.copy_(hidden)
        self.frame.copy_(sampled)
        self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
        self.position.add_(1)


@torch.inference_mode()
def generate_drama3_graphed(model, prompt, *, graphs, max_new_tokens, temperature, top_k, top_p,
                            prefill_chunk_size, frame_callback=None, stream_interval=64):
    """Fish's generate_drama3 with the decode step's audio decoder replayed from CUDA graphs.

    Prefill is unchanged. Returns CPU [frames, 1 + N] without EOS, and the finish reason.
    """
    if top_k < 1:
        raise InvalidInputError(f'top_k must be at least 1; got {top_k}')
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or capacity > graphs.capacity:
        raise ValueError(f'max_new_tokens={max_new_tokens} does not fit after {prompt_rows} prompt rows')
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=tower.device)
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)

    started = time.perf_counter()
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
    logger.info(f'Drama 3 prefill: {prompt_rows} rows in {time.perf_counter() - started:.2f}s')

    graphs.start(hidden[-1:], prompt_rows - 1, frames[:prompt_rows], temperature, top_k, top_p)
    produced, finish_reason = 0, 'length'
    for step in tqdm(range(max_new_tokens)):
        position = prompt_rows - 1 + step
        sampled, main, embeddings = graphs.step(position)
        if main == model.config.eos_token_id:
            finish_reason = 'eos'
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if frame_callback is not None and produced % stream_interval == 0:
            frame_callback(frames[prompt_rows: prompt_rows + produced].cpu())
        if step + 1 < max_new_tokens:
            hidden = backbone(inputs_embeds=embeddings[None],
                              input_pos=positions[position + 1: position + 2])[0].to(tower.device)
            graphs.advance(hidden, sampled)
    result = frames[prompt_rows: prompt_rows + produced].cpu()
    if frame_callback is not None and produced % stream_interval:
        frame_callback(result)
    return result, finish_reason
