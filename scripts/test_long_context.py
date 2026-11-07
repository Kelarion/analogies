"""
Checks the chunked SDPA scoring (ChatLM.score) on an interactive GPU node, before sweeping:

    python test_long_context.py [MODEL] [--lengths 4000 8000 16000 24000 32000]

1. which attention implementation the bridge left the model with, and memory after loading
2. correctness: on a short sequence, the new path (SDPA, chunked unembedding) against the
   old one (the bridge's eager forward, full logits): task log-probs and top tokens
3. memory: peak GPU memory of the new path at increasing lengths, until one runs out

Random token ids are fine for 3 (memory doesn't depend on content); 2 uses a real chat
prompt so the comparison is over a realistic distribution.
"""
import argparse
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch

from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='google/gemma-4-31B-it')
p.add_argument('--lengths', type=int, nargs='+', default=[4000, 8000, 16000, 24000, 32000])
p.add_argument('--check_len', type=int, default=1500,
               help='tokens in the correctness check (the old path must fit at this length)')
p.add_argument('--loose_kernels', action='store_true',
               help="allow SDPA's math fallback (only to diagnose an 'no available kernel' error)")
p.add_argument('--no_bypass', action='store_true',
               help="score through the bridge's wrappers (the behaviour before plain_hf)")
args = p.parse_args()

GB = 2 ** 30
lm = get_lm(args.model)
lm.strict_kernels = not args.loose_kernels
lm.bypass_bridge = not args.no_bypass
torch.cuda.synchronize()
print(f'loaded {args.model}')
print(f'  attention implementation(s) set by the bridge: {lm.attn_implementation}')
print(f'  allocated after loading: {torch.cuda.memory_allocated() / GB:.1f} GB '
      f'of {torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB')
head, cap = lm._unembed
print(f'  output layer: {type(head).__name__}, final logit soft-cap: {cap}')
with lm.plain_hf() as n:
    print(f'  bridge wrappers swapped out while scoring: {n if lm.bypass_bridge else 0}')

# -- 2. correctness ------------------------------------------------------------------------

text = ' '.join(f'The {a} walked past the {b} and stopped near the {c}.'
                for a, b, c in zip(*[np.random.default_rng(s).permutation(
                    ['cat', 'dog', 'river', 'tower', 'lamp', 'boat', 'tree', 'clock'] * 40)
                    for s in range(3)]))
prompt = lm.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                          add_generation_prompt=True, tokenize=False)
ids = lm.tokenizer(prompt, add_special_tokens=False).input_ids[:args.check_len]
ids = torch.as_tensor(ids, device=lm.device)
task_toks = np.unique(ids.cpu().numpy())[:20]          # tokens that actually occur

lp_old, ti_old, tl_old = lm.score_bridge(ids, task_toks)
torch.cuda.empty_cache()
lp_new, ti_new, tl_new = lm.score(ids, task_toks)

d = np.abs(lp_old - lp_new)
agree = (ti_old == ti_new).mean(0)
print(f'\ncorrectness on {len(ids)} tokens (old: bridge eager; new: SDPA + chunked unembed)')
print(f'  task log-prob |diff|: max {d.max():.4f}, mean {d.mean():.5f}, '
      f'99th pct {np.percentile(d, 99):.4f}')
print(f'  top-token agreement: full vocab {agree[0]:.3%}, text only {agree[1]:.3%}')
print(f'  top-token log-prob |diff|: max {np.abs(tl_old - tl_new).max():.4f}')
print('  differences are fine if they are no bigger than the model\'s own rounding noise\n'
      '  (diag_attention.py\'s noise floor: tiny for most models, large for Gemma 4).')

# does switching attention change anything? (bit-identical means the switch is ignored)
with torch.no_grad(), (lm.plain_hf() if lm.bypass_bridge else nullcontext()):
    with lm.attention('eager'):
        le = lm.hf_model(input_ids=ids[None], use_cache=False).logits[0]
    with lm.attention('sdpa'), lm._kernels():
        ls = lm.hf_model(input_ids=ids[None], use_cache=False).logits[0]
print(f'  eager vs sdpa logits, max |diff|: {(le.float() - ls.float()).abs().max().item():.3g} '
      '(0 means the attention setting is being ignored)')
del le, ls
torch.cuda.empty_cache()

# -- 3. memory -----------------------------------------------------------------------------

print('\npeak memory of score() by length')
lo, hi = 1000, len(lm.tokenizer) - 1000
for T in args.lengths:
    x = torch.randint(lo, hi, (T,), device=lm.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    try:
        lm.score(x, task_toks)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        print(f'  T={T:6d}: out of memory; where:')
        traceback.print_exc()
        break
    peak = torch.cuda.max_memory_allocated()
    print(f'  T={T:6d}: peak {peak / GB:5.1f} GB (+{(peak - base) / GB:4.1f} over weights), '
          f'{time.time() - t0:5.1f} s')
    del x