"""
Is the eager/SDPA gap a broken layer type, or the model's sensitivity to bf16 rounding?

    python diag_attention.py [MODEL] [--len 1500]

1. Every attention call, in an eager run and an SDPA run (flash/efficient kernels), checked
   against exact float32 attention on that call's own q, k, v with the intended mask
   (causal, plus the sliding window on sliding layers). A correct bf16 implementation is
   ~1e-3 to 1e-2 relative error; a broken layer type stands out by orders of magnitude.
2. The mask each implementation actually receives, for the first sliding and full layer.
3. How far the residual streams of the two runs have drifted apart after each layer.
4. A noise floor: eager vs eager on a batch of two copies (same maths, different matmul
   tiling, so only rounding changes). If this KL is as large as eager vs SDPA, the gap is
   bf16 sensitivity, not an SDPA bug.
"""
import argparse

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import transformers.models.gemma4.modeling_gemma4 as g4

from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='google/gemma-4-31B-it')
p.add_argument('--len', type=int, default=1500)
args = p.parse_args()

lm = get_lm(args.model)
hf = lm.hf_model

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
T = len(ids)
print(f'{T} tokens')


# -- probes ----------------------------------------------------------------------------------

def exact_attention(module, q, k, v, scaling):
    """float32 attention with the intended mask; (B, T, H, D) like HF's attention functions."""
    g = q.shape[1] // k.shape[1]
    kr, vr = k.repeat_interleave(g, 1).float(), v.repeat_interleave(g, 1).float()
    s = (q.float() @ kr.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scaling is None else scaling)
    i = torch.arange(q.shape[-2], device=q.device)[:, None]
    j = torch.arange(kr.shape[-2], device=q.device)[None]
    allow = j <= i
    if getattr(module, 'is_sliding', False):
        allow &= (i - j) < module.sliding_window
    return (s.masked_fill(~allow, float('-inf')).softmax(-1) @ vr).transpose(1, 2)


def mask_summary(mask):
    if mask is None:
        return 'None (SDPA uses is_causal)'
    m = mask[0, 0, :, :T] if mask.dim() == 4 else mask
    attend = m if m.dtype == torch.bool else (m > -1e3)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=m.device), 1)
    last = attend[-1].nonzero().flatten()
    return (f'{m.dtype} {tuple(mask.shape)}: attends to future={attend[future].any().item()}, '
            f'last row attends {len(last)} keys from position {last.min().item()}')


records = {}

def make_probe(fn, tag):
    def probe(module, q, k, v, mask, *a, **kw):
        out, w = fn(module, q, k, v, mask, *a, **kw)
        ref = exact_attention(module, q, k, v, kw.get('scaling'))
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        records.setdefault(tag, {})[module.layer_idx] = dict(
            sliding=module.is_sliding, hd=q.shape[-1], hq=q.shape[1], hkv=k.shape[1],
            rel=rel, mask=mask_summary(mask))
        return out, w
    return probe


@torch.no_grad()
def run(tag):
    """Hidden states after every layer, with attention probed. tag: 'eager' or 'sdpa'."""
    if tag == 'eager':
        orig = g4.eager_attention_forward
        g4.eager_attention_forward = make_probe(orig, tag)
        try:
            with lm.attention('eager'):
                out = hf.model(input_ids=ids[None], use_cache=False, output_hidden_states=True)
        finally:
            g4.eager_attention_forward = orig
    else:
        orig = ALL_ATTENTION_FUNCTIONS['sdpa']
        ALL_ATTENTION_FUNCTIONS['sdpa'] = make_probe(orig, tag)
        try:
            with lm.attention('sdpa'), lm._kernels():
                out = hf.model(input_ids=ids[None], use_cache=False, output_hidden_states=True)
        finally:
            ALL_ATTENTION_FUNCTIONS['sdpa'] = orig
    return [h[0] for h in out.hidden_states]


hs_e = run('eager')
torch.cuda.empty_cache()
hs_s = run('sdpa')
torch.cuda.empty_cache()

# -- 1 and 3: per layer ------------------------------------------------------------------------
E, S = records['eager'], records['sdpa']
print('\nlayer  type     hd  heads(q/kv)   eager err   sdpa err   resid drift after layer')
for l in sorted(E):
    e, s = E[l], S.get(l, {})
    drift = ((hs_e[l + 1].float() - hs_s[l + 1].float()).norm()
             / hs_e[l + 1].float().norm()).item() if l + 1 < len(hs_e) else float('nan')
    flag = '  <--' if max(e['rel'], s.get('rel', 0)) > 2e-2 else ''
    print(f'{l:5d}  {"slide" if e["sliding"] else "FULL ":7s} {e["hd"]:4d}  {e["hq"]:3d}/{e["hkv"]:<3d}'
          f'       {e["rel"]:9.2e}  {s.get("rel", float("nan")):9.2e}   {drift:9.2e}{flag}')

# -- 2: masks -----------------------------------------------------------------------------------
for tag, R in records.items():
    for kind in (True, False):
        l = min(l for l, r in R.items() if r['sliding'] == kind)
        print(f'\n{tag:5s} mask, first {"sliding" if kind else "full"} layer ({l}): {R[l]["mask"]}')

# -- 4: noise floor -------------------------------------------------------------------------------
del hs_e, hs_s
torch.cuda.empty_cache()

@torch.no_grad()
def kl_stats(x, y, chunk=256):
    kl, ag = [], []
    for a, b in zip(x.split(chunk), y.split(chunk)):
        la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        kl.append((la.exp() * (la - lb)).sum(-1))
        ag.append(la.argmax(-1) == lb.argmax(-1))
    kl, ag = torch.cat(kl).cpu().numpy(), torch.cat(ag).cpu().numpy()
    return f'mean KL {kl.mean():.2e}, max KL {kl.max():.2e}, top-1 agreement {ag.mean():.2%}'

with torch.no_grad(), lm.attention('eager'):
    l1 = hf(input_ids=ids[None], use_cache=False).logits[0]
    l2 = hf(input_ids=ids[None].repeat(2, 1), use_cache=False).logits[0]
with torch.no_grad(), lm.attention('sdpa'), lm._kernels():
    ls = hf(input_ids=ids[None], use_cache=False).logits[0]
print('\nnoise floor, eager bs=1 vs eager bs=2:', kl_stats(l1, l2))
print('for comparison, eager vs sdpa:        ', kl_stats(l1, ls))

print("""
Reading it:
  - one layer type (e.g. FULL, hd 512) with sdpa err >> eager err: SDPA is broken for those
    layers; the residual drift should jump at the first such layer.
  - a mask that attends to the future, or the wrong number of keys on the last row (sliding
    layers should see the window, full layers all T): a mask bug.
  - per-call errors all small in both, drift growing steadily, and a noise-floor KL about as
    large as eager vs sdpa: no bug; the model amplifies bf16 rounding, and eager is no more
    correct than SDPA.""")