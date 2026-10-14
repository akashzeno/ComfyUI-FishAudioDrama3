"""Full local Drama 3 job runtime; models are reused across chunks and variations."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# comfyenv mode keeps this runner in the repo and its state (Fish Speech checkout, model location,
# caches) in DRAMA3_STATE; the upstream isolated runtime keeps everything beside the runner.
STATE = Path(os.environ.get('DRAMA3_STATE') or ROOT)
(STATE / 'tmp').mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(STATE / 'tmp')
os.environ['TRITON_CACHE_DIR'] = str(STATE / 'triton-cache')
os.environ['FISH_DRAMA3_CUDA_GRAPH'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
compiler = Path(sys.prefix) / 'Lib/site-packages/triton/runtime/tcc/tcc.exe'
if compiler.exists():
    os.environ['CC'] = str(compiler)
sys.path.insert(0, str(STATE / 'fish-speech'))
try:
    import natsort  # noqa: F401  (Fish Speech imports it; only its file-listing helper uses it)
except ImportError:
    import types
    sys.modules['natsort'] = types.SimpleNamespace(natsorted=sorted)

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from fp8_offload import load_offloaded_model
from take_io import load_take, save_take
from fish_speech.models.dac.inference import load_model as load_codec
from fish_speech.models.dac.modded_dac import WindowLimitedTransformer
from fish_speech.models.text2semantic.fish_omni.chunking import Drama3History, PROTECTED_SPAN, parse_speaker_segments, split_drama3_text
from fish_speech.models.text2semantic.fish_omni.generate import Drama3PromptTooLongError, build_drama3_prompt, generate_drama3
from drama3_generate import DecodeGraphs, generate_drama3_graphed


FRAMES_PER_SECOND = 44100 / 2048  # Drama 3 codec: 44.1 kHz audio, 2048 samples per frame
FRAMES_PER_CHAR = 1.43            # measured: 112 spoken characters -> 160 frames
SOUND_SECONDS = .9                # rough length of one [sound] or [pause]
SOUND_WORDS = ('pause', 'laugh', 'sigh', 'breath', 'chuckle', 'giggle', 'gasp', 'cough', 'sniff', 'sob', 'cry', 'hum')
BRACKETS = re.compile(r'\[([^\]]*)\]')
ANGLE_TAGS = re.compile(r'<[^>]*>')


def estimate_frames(text):
    """Rough frame count of a part, used only to move the progress bar.

    Directions and word tags are not spoken; a short bracket naming a sound or pause is.
    """
    sounds = sum(1 for inner in BRACKETS.findall(text)
                 if len(inner.split()) <= 4 and any(word in inner.lower() for word in SOUND_WORDS))
    spoken = ' '.join(ANGLE_TAGS.sub('', BRACKETS.sub('', text)).split())
    return FRAMES_PER_CHAR * len(spoken) + sounds * SOUND_SECONDS * FRAMES_PER_SECOND


def generation_progress(variation, variants, part, parts, frames, estimate):
    """Overall progress while generating: 15 % (model loaded) to 95 % (saving).

    A part that runs past its estimate holds just short of the next part.
    """
    done = min(frames / max(estimate, 1.), .97)
    return .15 + .80 * (variation * parts + part + done) / (variants * parts)


def encode_reference(path, codec, cache=None):
    """Encode a reference clip. With a cache (persistent worker), each distinct clip is encoded once."""
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Reference audio must contain finite audio samples.')
    key = None
    if cache is not None:
        key = hashlib.sha1(np.ascontiguousarray(samples).tobytes() + str(rate).encode()).hexdigest()
        if key in cache:
            return cache[key]
    codes = encode_samples(samples.mean(axis=1), rate, codec)
    if cache is not None:
        cache[key] = codes
    return codes


def encode_samples(mono, rate, codec):
    wave = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32)).unsqueeze(0)
    wave = torchaudio.functional.resample(wave, rate, codec.sample_rate)
    with torch.inference_mode():
        audio = wave.unsqueeze(0).cuda()
        indices, _ = codec.encode(audio, torch.tensor([audio.shape[-1]], device='cuda'))
    return indices[0].cpu().numpy().astype(np.int16)


def decode_window(codec, codes, start, end):
    begin = max(0, start - 128)
    tensor = torch.from_numpy(np.ascontiguousarray(codes[:, begin:end])).long().unsqueeze(0).cuda()
    with torch.inference_mode():
        audio = codec.from_indices(tensor)[0, 0].float().cpu().numpy()
    audio = audio[(start - begin) * 2048:]
    if not np.isfinite(audio).all():
        raise FloatingPointError('Non-finite audio samples.')
    return audio


def directed_text(text, direction):
    if not direction.strip():
        return text
    cue = direction.strip()
    if not cue.startswith('['):
        cue = '[' + cue + ']'
    parts = re.split(r'(<\|speaker:\d+\|>)', text)
    if parts[0].strip():
        parts[0] = cue + ' ' + parts[0]
    for i in range(2, len(parts), 2):
        parts[i] = ' ' + cue + ' ' + parts[i].lstrip()
    return ''.join(parts)


def split_clauses(text):
    """Keep every clause queued, protecting cues, speaker tags and paired tags."""
    tagged = bool(re.search(r'<\|speaker:\d+\|>', text))
    result = []
    for speaker, body in parse_speaker_segments(text):
        spans = iter(PROTECTED_SPAN.finditer(body))
        protected = next(spans, None)
        start, index = 0, 0
        while index < len(body):
            if protected is not None and index == protected.start():
                index = protected.end()
                protected = next(spans, None)
                continue
            decimal = body[index] == '.' and 0 < index < len(body)-1 and body[index-1].isdigit() and body[index+1].isdigit()
            if body[index] in ',;:。.!?，；：！？\n' and not decimal:
                piece = body[start:index+1].strip()
                if piece:
                    result.append((f'<|speaker:{speaker}|> ' if tagged else '') + piece)
                start = index+1
            index += 1
        tail = body[start:].strip()
        if tail:
            result.append((f'<|speaker:{speaker}|> ' if tagged else '') + tail)
    return result


def build_speaker_prompt(chunk, tokenizer, config, voices):
    segments = parse_speaker_segments(chunk)
    if len(segments) != 1:
        raise ValueError('Speaker mode requires one speaker per part.')
    speaker, _ = segments[0]
    references = [ref for ref in voices if ref['speaker_id'] == int(speaker)]
    return build_drama3_prompt(chunk, tokenizer, config,
        [f"<|speaker:{speaker}|> {ref['text']}" for ref in references],
        [torch.from_numpy(ref['codes']).long() for ref in references])


# Fast mode reads the prompt in 512-row chunks instead of 32. Each chunk routes to most of every layer's
# experts, all copied from RAM: 32-row chunks moved about 180 GB over PCIe for a 400-row prompt.
FAST_PREFILL_ROWS = 512
CODEC_RESERVE_GIB = 3.0  # codec beside the model: 1.6 GB of weights + up to 0.95 GB while encoding


CODEC_MASK_SIZE = 32768  # modded_dac's Transformer precomputes a causal mask this many rows square


@contextlib.contextmanager
def unused_codec_masks_skipped():
    """While the codec is built, make its 32768 x 32768 causal masks empty placeholders.

    Building them is torch.tril over 1 GiB each on the CPU: 15 of the codec's 22 s of loading,
    for masks load_codec_slim then drops. Every other torch.tril call is untouched.
    """
    real_tril = torch.tril

    def tril(input, *args, **kwargs):
        if not args and not kwargs and input.dtype == torch.bool and input.shape == (CODEC_MASK_SIZE, CODEC_MASK_SIZE):
            return input.new_empty(0)
        return real_tril(input, *args, **kwargs)

    torch.tril = tril
    try:
        yield
    finally:
        torch.tril = real_tril


def load_codec_slim(checkpoint):
    """Load the audio codec without its unused 32768 x 32768 attention masks (1 GiB each).

    The masks belong to WindowLimitedTransformer modules, whose forward always builds its own mask
    and passes it in; the precomputed one is read only when no mask is passed, which never happens
    there. Dropping it changes no value and cuts the codec from about 4.6 GB to 1.6 GB of VRAM; not
    building it at all saves 15 s of loading.
    """
    with unused_codec_masks_skipped():
        codec = load_codec('modded_dac_vq', str(checkpoint / 'codec.pth'), device='cpu')
    for module in codec.modules():
        if isinstance(module, WindowLimitedTransformer):
            module.causal_mask = None
        elif getattr(module, 'causal_mask', None) is not None and module.causal_mask.numel() == 0:
            raise RuntimeError(f'{type(module).__name__} needs its causal mask, but it was skipped')
    return codec.cuda()


def checkpoint_info():
    checkpoint = Path(json.loads((STATE / 'model-location.json').read_text(encoding='utf-8-sig'))['checkpoint_dir'])
    return checkpoint, (checkpoint / 'revision.txt').read_text().strip()


def load_state(budget):
    """Load the codec and the model once, for a persistent worker.

    The codec stays on the GPU beside the model, so the model's budget leaves CODEC_RESERVE_GIB free.
    That only moves the split between resident and staged experts, which compute the same values.
    """
    checkpoint, revision = checkpoint_info()
    codec = load_codec_slim(checkpoint)
    # The first encode in a process takes slightly different floating-point paths than every later
    # one (equally accurate: the same reconstruction error, but different codes). A one-job process
    # always encodes its reference first, so the worker caches each reference's codes: the first
    # reference it meets is encoded exactly as a one-job process would, and every later job with
    # that reference reuses those codes. Its takes then match the one-job mode bit for bit.
    model = load_offloaded_model(checkpoint, gpu_budget_gib=budget - CODEC_RESERVE_GIB)
    graphs = DecodeGraphs.build(model)  # before any seed: capturing runs the sampler
    return {'checkpoint': checkpoint, 'revision': revision, 'codec': codec, 'model': model, 'references': {},
            'graphs': graphs}


def run_job(job, output, budget, state=None):
    """One job. With state (a persistent worker) the loaded model and codec are reused and kept."""
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint, revision = (state['checkpoint'], state['revision']) if state else checkpoint_info()
    events = output.parent / 'events.jsonl'
    events.write_text('', encoding='utf-8')

    def emit(**event):
        with events.open('a', encoding='utf-8') as file:
            file.write(json.dumps(event) + '\n')

    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    refs = job.get('voices', [])
    previous = load_take(job['previous_take']) if job.get('previous_take') else {'turns': [], 'voices': []}
    if previous.get('revision', revision) != revision:
        raise ValueError('The previous take belongs to another checkpoint revision.')
    streaming = job.get('streaming', False)
    codec = state['codec'] if state else None
    if refs or (job.get('previous_audio') and job.get('use_history', True)) or streaming:
        emit(stage='Encoding references' if state else 'Encoding references / loading audio codec', progress=.02)
        if codec is None:
            codec = load_codec_slim(checkpoint)
    replaced = {ref['speaker_id'] for ref in refs}
    voices = [ref for ref in previous['voices'] if ref['speaker_id'] not in replaced]
    for ref in refs:
        if not ref['text'].strip():
            raise ValueError('Every reference needs its exact transcript.')
        voices.append({'text': ref['text'], 'speaker_id': ref['speaker_id'],
                       'codes': encode_reference(ref['path'], codec, state['references'] if state else None)})
    if job.get('previous_audio') and job.get('use_history', True):
        previous['turns'].append({'text': job['previous_text'], 'codes': encode_reference(job['previous_audio'], codec)})
    if codec is not None and not streaming and not state:
        del codec
        codec = None
        gc.collect()
        torch.cuda.empty_cache()
    encoded = time.perf_counter()
    if state:
        model = state['model']
        graphs = state.get('graphs')
        pool_before = model.offload_pool.stats()
    else:
        emit(stage='Loading original FP8 model (most of it into RAM)', progress=.06)
        model = load_offloaded_model(checkpoint, gpu_budget_gib=budget - (CODEC_RESERVE_GIB if streaming else 0))
        graphs = DecodeGraphs.build(model)  # before any seed: capturing runs the sampler
        pool_before = {}
    loaded = time.perf_counter()
    text = job['text']
    mode = job.get('chunk_mode', 'auto')
    lines = ([f'<|speaker:{speaker}|> {body.strip()}' for speaker, body in parse_speaker_segments(text)]
             if mode == 'speakers' else split_clauses(text) if mode == 'clauses' else text.splitlines() if mode == 'lines' else [text])
    chunks = [text] if mode == 'single' else [part for line in lines for part in split_drama3_text(line, max_bytes=job.get('chunk_bytes', 300))]
    if not chunks or not text.strip():
        raise ValueError('Enter some text to speak.')
    chunks = [directed_text(chunk, job.get('direction', '')) for chunk in chunks]
    prompt_texts = [f"<|speaker:{ref['speaker_id']}|> {ref['text']}" for ref in voices]
    prompt_codes = [torch.from_numpy(ref['codes']).long() for ref in voices]
    results = []
    variants = job.get('variations', 1)
    for variation in range(variants):
        seed = (job.get('seed', 42) + variation) % 2**64
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        history = Drama3History(recent_turns=job.get('history_turns', 1))
        if job.get('use_history', True):
            for turn in previous['turns']:
                history.push(turn['text'], torch.from_numpy(turn['codes']).long())
        turns, samples, finishes = [], [], []
        for index, chunk in enumerate(chunks):
            estimate = estimate_frames(chunk)
            emit(stage=f'Take {variation+1}/{variants} - part {index+1}/{len(chunks)} - starting',
                 progress=generation_progress(variation, variants, index, len(chunks), 0, estimate))
            for candidates in ([()] if mode == 'speakers' else history.candidates()):
                try:
                    prompt = (build_speaker_prompt(chunk, model.tokenizer, model.config, voices) if mode == 'speakers' else
                              build_drama3_prompt(chunk, model.tokenizer, model.config, prompt_texts, prompt_codes, history=candidates))
                    break
                except Drama3PromptTooLongError as exc:
                    error = exc
            else:
                raise error
            decoded = 0
            part_samples = []

            def preview(frames):
                nonlocal decoded
                codes = frames[:, 1:].T.contiguous().numpy()
                audio = decode_window(codec, codes, decoded, len(frames))
                decoded = len(frames)
                part_samples.append(audio)
                filename = f'preview_{variation}_{index}_{len(part_samples)}.wav'
                sf.write(output.parent / filename, audio, codec.sample_rate, subtype='PCM_16')
                emit(stage=f'Take {variation+1} - part {index+1} - {len(frames)} frames', audio=filename, variation=variation,
                     progress=generation_progress(variation, variants, index, len(chunks), len(frames), estimate))

            def report(frames):
                emit(stage=f'Take {variation+1}/{variants} - part {index+1}/{len(chunks)} - '
                           f'{len(frames) / FRAMES_PER_SECOND:.1f} s of audio so far',
                     progress=generation_progress(variation, variants, index, len(chunks), len(frames), estimate))

            limit = job.get('max_frames', 0) or (600 if len(chunks) > 1 else 2048)
            sampling = dict(max_new_tokens=limit, prefill_chunk_size=32, temperature=job.get('temperature', .7),
                            top_p=job.get('top_p', .7), top_k=job.get('top_k', 30),
                            frame_callback=preview if streaming else report, stream_interval=64 if streaming else 16)
            if graphs is not None and graphs.fits(len(prompt) + limit):
                frames, finish = generate_drama3_graphed(model, prompt, graphs=graphs,
                                                         **(sampling | {'prefill_chunk_size': FAST_PREFILL_ROWS}))
            else:
                frames, finish = generate_drama3(model, prompt, **sampling)
            if not len(frames):
                raise RuntimeError(f'No audio frames produced for part {index+1}.')
            codes = frames[:, 1:].T.contiguous().numpy().astype(np.int16)
            history.push(chunk, torch.from_numpy(codes).long())
            turns.append({'text': chunk, 'codes': codes})
            finishes.append(finish)
            samples.extend(part_samples)
            if finish == 'length':
                emit(stage='Frame limit reached: increase max_frames if speech is cut short.')
        results.append({'turns': turns, 'samples': samples, 'seed': seed, 'finish_reasons': finishes})
    torch.cuda.synchronize()
    generated = time.perf_counter()
    pool = {key: value - pool_before.get(key, 0) if key != 'staging_capacity_experts' else value
            for key, value in model.offload_pool.stats().items()}
    stats = pool | {'persistent_worker': bool(state), 'reference_encode_seconds': encoded-started, 'model_load_seconds': loaded-encoded,
        'generation_seconds': generated-loaded, 'generation_peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
        'generation_peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30, 'process_RAM_GiB': psutil.Process().memory_info().rss/2**30,
        'checkpoint_revision': revision, 'reference_count': len(voices), 'parts': len(chunks), 'chunk_mode': mode,
        'part_speakers': [speaker for chunk in chunks for speaker, _ in parse_speaker_segments(chunk)],
        'streaming': streaming, 'text': job['text']}
    if not state:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    emit(stage='Saving audio and reusable takes', progress=.96)
    if codec is None:
        codec = load_codec_slim(checkpoint)
    files = []
    for variation, result in enumerate(results):
        if not streaming:
            for turn in result['turns']:
                codes = turn['codes']
                result['samples'].extend(decode_window(codec, codes, start, min(start+256, codes.shape[1])) for start in range(0, codes.shape[1], 256))
        audio = np.concatenate(result.pop('samples'))
        path = output if variation == 0 else output.with_name(f'{output.stem}_{variation+1}{output.suffix}')
        sf.write(path, audio, codec.sample_rate, subtype='PCM_16')
        take_path = path.with_suffix('.npz')
        save_take(take_path, {'revision': revision, 'voices': voices,
            'turns': previous['turns'] + result['turns'] if job.get('use_history', True) else result['turns']})
        files.append({'audio': path.name, 'take': take_path.name, 'seed': result['seed'], 'finish_reasons': result['finish_reasons'],
            'audio_seconds': len(audio)/codec.sample_rate, 'frames': sum(turn['codes'].shape[1] for turn in result['turns'])})
    stats.update(results=files, total_seconds=time.perf_counter()-started, sample_rate=codec.sample_rate)
    output.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    emit(stage='Done', progress=1.)
    print(json.dumps(stats, indent=2), flush=True)


def serve(server_dir, budget, idle_seconds, parent_pid=None):
    """Persistent worker: load once, run queued jobs, exit when idle, asked to stop, or ComfyUI exits."""
    import worker_protocol as wp
    server_dir = Path(server_dir)
    started = time.perf_counter()
    print('[Drama 3 worker] loading the codec and the model ...', flush=True)
    state = load_state(budget)
    load_seconds = time.perf_counter() - started
    wp.mark_ready(server_dir, {'pid': os.getpid(), 'load_seconds': load_seconds, 'gpu_budget': budget})
    print(f'[Drama 3 worker] ready in {load_seconds:.0f} s; exits after {idle_seconds / 60:.0f} idle minutes', flush=True)
    last = time.monotonic()
    while True:
        if wp.stop_requested(server_dir):
            reason = 'asked to stop'
            break
        if parent_pid and not psutil.pid_exists(parent_pid):
            reason = 'ComfyUI exited'
            break
        claimed = wp.claim_next(server_dir)
        if claimed is None:
            if time.monotonic() - last > idle_seconds:
                reason = f'idle for {idle_seconds / 60:.0f} minutes'
                break
            time.sleep(.2)
            continue
        job_id, payload = claimed
        try:
            job = json.loads(Path(payload['request']).read_text(encoding='utf-8'))
            run_job(job, Path(payload['output']), budget, state=state)
            wp.finish(server_dir, job_id, ok=True)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            wp.finish(server_dir, job_id, ok=False, error=error)
        torch.cuda.empty_cache()
        last = time.monotonic()
    print(f'[Drama 3 worker] exiting ({reason})', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path)
    parser.add_argument('--text', default='Hello, this is a local test.')
    parser.add_argument('--max-frames', type=int, default=96)
    parser.add_argument('--gpu-budget', type=float, default=26.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-audio', type=Path)
    parser.add_argument('--reference-text', default='')
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.7)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--output', type=Path, default=STATE.parent/'outputs/drama3-fp8-test.wav')
    parser.add_argument('--serve', type=Path, help='Run as a persistent worker serving jobs from this folder.')
    parser.add_argument('--idle-seconds', type=float, default=600.)
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve, args.gpu_budget, args.idle_seconds, args.parent_pid)
        return
    job = json.loads(args.request.read_text(encoding='utf-8')) if args.request else {
        'text': args.text, 'seed': args.seed, 'max_frames': args.max_frames, 'temperature': args.temperature,
        'top_p': args.top_p, 'top_k': args.top_k, 'voices': [{'path': str(args.reference_audio), 'text': args.reference_text, 'speaker_id': 0}] if args.reference_audio else []}
    run_job(job, args.output, args.gpu_budget)


if __name__ == '__main__':
    main()
