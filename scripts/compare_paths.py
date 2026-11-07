"""
Is the gap between the bridge's forward and plain HF rounding noise, or a different computation?

    python compare_paths.py MODEL PROMPT_FILE [--len 3000]

PROMPT_FILE: one decoded prompt from a saved experiment (metrics['prompts']), as plain text.
Logits for the same tokens, five ways:
  bridge        the bridge's forward, wrappers and all (the old scoring path)
  eager, eager2 plain HF (bridge wrappers swapped out), eager attention, batch of 1 and 2
  sdpa, sdpa2   plain HF, SDPA attention, batch of 1 and 2
eager vs eager2 is the noise floor: the same maths with different matmul tiling, so only the
rounding changes. (sdpa vs sdpa2 can come out exactly 0; that only means SDPA's rounding
doesn't depend on batch size, not that the model is stable.)
"""
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('prompt_file')
p.add_argument('--len', type=int, default=3000)
args = p.parse_args()

lm = get_lm(args.model)
text = open(args.prompt_file).read()
all_ids = lm.tokenizer(text, add_special_tokens=False).input_ids
ids = torch.as_tensor(all_ids[:args.len], device=lm.device)
print(f'prompt: {len(all_ids)} tokens, using the first {len(ids)}')


@torch.no_grad()
def plain(impl, bs=1):
    with lm.plain_hf(), lm.attention(impl), (lm._kernels() if impl == 'sdpa' else nullcontext()):
        return lm.hf_model(input_ids=ids[None].repeat(bs, 1), use_cache=False).logits[0]


with torch.no_grad():
    L = {'bridge': lm.bridge(ids[None])[0]}
torch.cuda.empty_cache()
for name, impl, bs in [('eager', 'eager', 1), ('eager2', 'eager', 2),
                       ('sdpa', 'sdpa', 1), ('sdpa2', 'sdpa', 2)]:
    L[name] = plain(impl, bs)
    torch.cuda.empty_cache()


@torch.no_grad()
def compare(x, y, chunk=256):
    kl, agree, margin, dmax = [], [], [], []
    for a, b in zip(x.split(chunk), y.split(chunk)):
        la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        kl.append((la.exp() * (la - lb)).sum(-1))
        agree.append(la.argmax(-1) == lb.argmax(-1))
        t2 = la.topk(2, -1).values
        margin.append(t2[:, 0] - t2[:, 1])
        dmax.append(((la - lb).abs() * (la > np.log(1e-4))).amax(-1))
    return [torch.cat(v).cpu().numpy() for v in (kl, agree, margin, dmax)]


print('\n                     mean KL    max KL   agree  agree(margin>1)  max|dlogp| (p>1e-4)')
for x, y, note in [('eager', 'eager2', 'noise floor (eager)'),
                   ('sdpa', 'sdpa2', 'noise floor (sdpa; 0 is uninformative)'),
                   ('eager', 'sdpa', 'attention implementation'),
                   ('bridge', 'eager', 'bridge vs plain HF'),
                   ('bridge', 'sdpa', 'old scoring vs new')]:
    kl, ag, mg, dm = compare(L[x], L[y])
    print(f'  {x:>6s} vs {y:<6s}  {kl.mean():9.2e} {kl.max():9.2e} {ag.mean():7.2%} '
          f'{ag[mg > 1].mean():12.2%}      {dm.max():9.3f}     {note}')

print("""
Reading it:
  - bridge vs eager about the size of the eager noise floor: the bridge computes the same
    thing as HF, just expensively. Old and new scoring differ only by rounding.
  - bridge vs eager well above the noise floor, while eager vs sdpa is near it: the bridge
    computes this model's attention differently from HF, so results scored through the
    bridge are suspect, not just noisy.
  - eager vs sdpa well above the noise floor: HF's SDPA path for this model is the problem.""")