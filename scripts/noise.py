import sys, torch
from icl_model import get_lm

lm = get_lm(sys.argv[1]); hf = lm.hf_model
text = open(sys.argv[2]).read()              # one decoded prompt from metrics['prompts']
ids = torch.as_tensor(lm.tokenizer(text, add_special_tokens=False).input_ids[:3000],
                      device=lm.device)
with torch.no_grad():
    with lm.attention('eager'):
        l1 = hf(input_ids=ids[None], use_cache=False).logits[0].float().log_softmax(-1)
    with lm.attention('sdpa'), lm._kernels():
        l2 = hf(input_ids=ids[None], use_cache=False).logits[0].float().log_softmax(-1)
# with torch.no_grad(), lm.attention('sdpa'), lm._kernels():
#     l1 = hf(input_ids=ids[None], use_cache=False).logits[0].float().log_softmax(-1)
#     l2 = hf(input_ids=ids[None].repeat(2, 1), use_cache=False).logits[0].float().log_softmax(-1)
kl = (l1.exp() * (l1 - l2)).sum(-1)
print(f'{len(ids)} tokens: mean KL {kl.mean():.2e}, max {kl.max():.2e}, '
      f'top-1 agreement {(l1.argmax(-1) == l2.argmax(-1)).float().mean():.2%}') 