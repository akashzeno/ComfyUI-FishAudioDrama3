"""Stand-in for `run_drama3_full.py --serve`: same protocol, no model. Used by test_worker_manager."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import worker_protocol as wp

parser = argparse.ArgumentParser()
parser.add_argument('--serve', type=Path, required=True)
parser.add_argument('--gpu-budget', type=float)
parser.add_argument('--idle-seconds', type=float, default=600)
parser.add_argument('--parent-pid', type=int)
args = parser.parse_args()
print('fake worker loading', flush=True)
time.sleep(.3)
wp.mark_ready(args.serve, {'pid': 0})
last = time.monotonic()
while not wp.stop_requested(args.serve):
    claimed = wp.claim_next(args.serve)
    if claimed is None:
        if time.monotonic() - last > args.idle_seconds:
            print('fake worker idle exit', flush=True)
            break
        time.sleep(.05)
        continue
    job_id, payload = claimed
    text = json.loads(Path(payload['request']).read_text(encoding='utf-8'))['text']
    if text == 'die':
        sys.exit(3)
    if text == 'slow':
        time.sleep(30)
    if text == 'fail':
        wp.finish(args.serve, job_id, ok=False, error='boom')
    else:
        Path(payload['output']).write_text(text, encoding='utf-8')
        wp.finish(args.serve, job_id, ok=True)
    print(f'fake worker did {text}', flush=True)
    last = time.monotonic()
