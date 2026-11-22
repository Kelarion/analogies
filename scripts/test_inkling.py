"""
Checks the blocked attention for Inkling 2 (chunked_attention.py) on a GPU node:

    python test_inkling.py [--prompt PROMPT_FILE] [--len 2000] [--lengths 4000 8000 12000 16000]

1. per call: every self-attention call of one forward pass, blocked vs the original eager
   code on identical inputs (relative error; rounding is ~1e-3 or below)
2. end to end: final distributions, blocked vs the original eager forward
3. memory: peak GPU memory of score() at increasing lengths, with the traceback on OOM

--len should be over 1024 so the sliding window is exercised; the original eager code has to
fit at that length (it does at 2000). PROMPT_FILE: one decoded prompt from a saved experiment;
without it, a stock paragraph.
"""
import argparse
import time
import traceback

import numpy as np
import torch

import chunked_attention as ca
from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='thinkingmachines/Inkling-2-Mini')
p.add_argument('--prompt')
p.add_argument('--len', type=int, default=2000)
p.add_argument('--lengths', type=int, nargs='+', default=[4000, 8000, 12000, 16000])
p.add_argument('--block', type=int, default=None,
               help='queries per block in checks 1-2 (default: automatic)')
args = p.parse_args()

GB = 2 ** 30
lm = get_lm(args.model)
hf = lm.hf_model
cfg = hf.config.get_text_config() if hasattr(hf.config, 'get_text_config') else hf.config
print(f'loaded {args.model}: scoring attention = {lm.scoring_attention}')
print(f'  allocated after loading: {torch.cuda.memory_allocated() / GB:.1f} GB')
for k in ('sliding_window_size', 'rel_extent', 'num_attention_heads', 'swa_num_attention_heads',
          'num_key_value_heads', 'swa_num_key_value_heads', 'head_dim', 'swa_head_dim'):
    print(f'  {k}: {getattr(cfg, k, "-")}')

if args.prompt:
    text = open(args.prompt).read()
else:
    text = ("The history of cartography is the history of people trying to fit a curved world "
            "onto flat surfaces. Early maps were as much about belief as geography, and every "
            "projection traded one distortion for another. ") * 80
ids = torch.as_tensor(lm.tokenizer(text, add_special_tokens=False).input_ids[:args.len],
                      device=lm.device)
print(f'\n{len(ids)} tokens for checks 1-2')
ca.QUERY_BLOCK = args.block

# -- 1. every self-attention call against the original code ----------------------------------
with torch.no_grad(), lm.plain_hf(), lm.attention('eager'), \
        ca.inkling2_chunked_attention(check=True) as records:
    hf.model(input_ids=ids[None], use_cache=False)
errs = np.array([r['rel_err'] for r in records])
sliding = np.array([bool(r['sliding']) for r in records])
print(f'\n1. per call: {len(records)} self-attention calls '
      f'({sliding.sum()} sliding, {(~sliding).sum()} full)')
for name, sel in (('sliding', sliding), ('full', ~sliding)):
    if sel.any():
        print(f'  {name:7s} relative error vs original: max {errs[sel].max():.2e}, '
              f'median {np.median(errs[sel]):.2e}')
worst = int(errs.argmax())
print(f'  worst call: #{worst} (layer {records[worst]["layer"]}); '
      f'{"OK" if errs.max() < 1e-2 else "TOO LARGE: not rounding, send me this output"}')
torch.cuda.empty_cache()

# -- 2. end to end -------------------------------------------------------------------------------
with torch.no_grad():
    with lm.plain_hf(), lm.attention('eager'):
        h_ref = hf.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    torch.cuda.empty_cache()
    h_new = lm.hidden_states(ids)
    kl, agree = [], []
    for a, b in zip(h_ref.split(256), h_new.split(256)):
        la = torch.log_softmax(lm.logits_from_hidden(a).float(), -1)
        lb = torch.log_softmax(lm.logits_from_hidden(b).float(), -1)
        kl.append((la.exp() * (la - lb)).sum(-1))
        agree.append(la.argmax(-1) == lb.argmax(-1))
    kl, agree = torch.cat(kl).cpu().numpy(), torch.cat(agree).cpu().numpy()
print(f'\n2. end to end, blocked vs original: final hidden state relative diff '
      f'{((h_new.float() - h_ref.float()).norm() / h_ref.float().norm()).item():.2e}, '
      f'mean KL {kl.mean():.2e}, max KL {kl.max():.2e}, top-1 agreement {agree.mean():.2%}')
print('  (rounding differences grow through the layers, as with Gemma and Qwen; check 1 is the\n'
      '  one that says whether the blocked code computes the same thing)')
del h_ref, h_new
torch.cuda.empty_cache()

# -- 3. memory -----------------------------------------------------------------------------------
ca.QUERY_BLOCK = None
print('\n3. peak memory of score() by length')
task_toks = np.unique(ids.cpu().numpy())[:20]
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