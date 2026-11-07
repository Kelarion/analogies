"""
Where do the old (bridge, eager) and new (SDPA, chunked unembedding) scoring paths diverge?

    python diag_scoring.py [MODEL] [--len 1500]

Logits for one prompt, five ways, each pair below changing one thing:
  A  bridge forward (the old run_trial)
  B  HF forward, eager attention                       A-B: the bridge vs plain HF
  M  HF forward, SDPA restricted to the 'math' kernel  B-M: HF's SDPA code path itself
                                                            (math is numerically ~eager, so a
                                                            gap here is semantics: an ignored
                                                            soft-cap, a different mask)
  C  HF forward, SDPA flash/efficient kernels          M-C: kernel numerics alone
  D  ChatLM.score's path: hidden states + own unembed  C-D: my unembedding vs HF's
Plus A-D, which is what test_long_context.py measured.
"""
import argparse
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='google/gemma-4-31B-it')
p.add_argument('--len', type=int, default=1500)
args = p.parse_args()

lm = get_lm(args.model)
hf = lm.hf_model

# -- the config fields that could make eager and SDPA differ -------------------------------
cfg = hf.config
tc = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
print('text config fields about soft-capping, windows, scaling, head sizes:')
for k, v in sorted(tc.to_dict().items()):
    if any(s in k for s in ('softcap', 'window', 'scal', 'head_dim', 'layer_types', 'attn')):
        print(f'  {k}: {dict(Counter(v)) if isinstance(v, list) else v}')

# -- a natural-language prompt (fewer near-ties than the word list in the first test) --------
text = ("The history of cartography is the history of people trying to fit a curved world "
        "onto flat surfaces. Early maps were as much about belief as geography: medieval "
        "mappae mundi placed Jerusalem at the centre and filled the margins with monsters. "
        "With the arrival of printing, careful surveying, and later satellites, maps became "
        "tools of navigation, taxation, and war, and every projection traded one distortion "
        "for another. ") * 30
prompt = lm.tokenizer.apply_chat_template(
    [{'role': 'user', 'content': text + '\nSummarise this in one paragraph.'}],
    add_generation_prompt=True, tokenize=False)
ids = torch.as_tensor(lm.tokenizer(prompt, add_special_tokens=False).input_ids[:args.len],
                      device=lm.device)
print(f'\n{len(ids)} tokens')


@torch.no_grad()
def hf_logits(impl, backends=None):
    with lm.attention(impl), (sdpa_kernel(backends) if backends else nullcontext()):
        return hf(input_ids=ids[None], use_cache=False).logits[0]


with torch.no_grad():
    L = {}
    L['A'] = lm.bridge(ids[None])[0]
    L['B'] = hf_logits('eager')
    L['M'] = hf_logits('sdpa', [SDPBackend.MATH])
    L['C'] = hf_logits('sdpa', [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    L['D'] = torch.cat([lm.logits_from_hidden(hc) for hc in lm.hidden_states(ids).split(512)])
    torch.cuda.empty_cache()


@torch.no_grad()
def compare(x, y, chunk=256):
    """Per position: KL(x||y), whether the argmax agrees, x's top-1 margin, and the largest
    |log p_x - log p_y| over tokens x gives at least 1e-4."""
    kl, agree, margin, dmax = [], [], [], []
    for a, b in zip(x.split(chunk), y.split(chunk)):
        la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        kl.append((la.exp() * (la - lb)).sum(-1))
        agree.append(la.argmax(-1) == lb.argmax(-1))
        t2 = la.topk(2, -1).values
        margin.append(t2[:, 0] - t2[:, 1])
        dmax.append(((la - lb).abs() * (la > np.log(1e-4))).amax(-1))
    return [torch.cat(v).cpu().numpy() for v in (kl, agree, margin, dmax)]


bins = np.linspace(0, len(ids), 7).astype(int)
print('\n          mean KL    max KL   agree  agree(margin>1)  max|dlogp| (p>1e-4)'
      '   mean KL by position: ' + ' '.join(f'{b0}-{b1}' for b0, b1 in zip(bins, bins[1:])))
for x, y in [('A', 'B'), ('B', 'M'), ('M', 'C'), ('C', 'D'), ('A', 'D')]:
    kl, ag, mg, dm = compare(L[x], L[y])
    conf = mg > 1
    by_pos = ' '.join(f'{kl[b0:b1].mean():.3g}' for b0, b1 in zip(bins, bins[1:]))
    print(f'  {x}-{y}:  {kl.mean():9.2e} {kl.max():9.2e} {ag.mean():7.2%} '
          f'{ag[conf].mean():12.2%}      {dm.max():9.3f}            {by_pos}')

print("""
Reading it: bf16 noise between two correct paths is roughly mean KL <~1e-3 and agreement
~99%+ where the margin is > 1 nat. Any pair well above that is where the difference comes in:
  A-B large  the bridge's forward differs from plain HF (the old path was the odd one out)
  B-M large  HF's SDPA path computes something different from eager (look above for an
             attention soft-cap, or a gap that starts at the sliding window in the
             by-position column)
  M-C large  the flash/efficient kernel itself (unusual)
  C-D large  my unembedding (soft-cap or scaling) doesn't match HF's""")