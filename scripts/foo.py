import numpy as np, torch
from icl_model import get_lm

lm = get_lm('thinkingmachines/Inkling-2-Mini')
ids = torch.tensor(np.random.default_rng(0).integers(1000, min(len(lm.tokenizer), 100_000), 3000),
                   device=lm.device)
with torch.no_grad(), lm.plain_hf():
    lp = [torch.log_softmax(lm.logits_from_hidden(lm._hidden_states_blocked(ids, b)).float(), -1)
          for b in (None, 1024)]
d = (lp[0] - lp[1]).abs().max(-1).values.cpu().numpy()
kl = (lp[0].exp() * (lp[0] - lp[1])).sum(-1).cpu().numpy()
top = (lp[0].argmax(-1) == lp[1].argmax(-1)).cpu().numpy()
for s in range(0, len(ids), 1024):
    e = min(s + 1024, len(ids))
    print(f'positions {s:4d}-{e:4d}: median max|dlogp| {np.median(d[s:e]):.3f}, '
          f'mean KL {kl[s:e].mean():.2e}, same top token {top[s:e].mean():.1%}')