"""
Exact, memory-efficient attention for models transformers can only run with eager attention.

Inkling 2 (transformers.models.inkling2) declares no SDPA / flash / flex support, because its
attention adds a learned per-token relative position bias and per-token sinks. Its eager path
builds, for every attention call, a (q_len, kv_len) distance matrix, a (heads, q_len, kv_len)
bias, the scores, and a float32 softmax over scores plus a sink column: memory quadratic in
length, tens of GB per call at ~10k tokens.

inkling2_chunked_attention() patches three things while it is active:
  get_relative_distance            returns a LazyDistance, which builds any slice on demand
  Inkling2RelativePositionBias     returns a LazyBias: the per-token relative states (linear
                                   in length), turned into bias values one slice at a time
  eager_attention_forward          for the self-attention calls (the ones given a LazyBias),
                                   runs a block of queries at a time with the same operations
                                   as the eager code, over only the keys the block can attend
                                   to (causal; within the window on sliding layers). Every other
                                   call (the block-attention residuals) goes to the original.
Peak memory is then about one block's worth (QUERY_BLOCK queries x keys x heads).

It only takes over calls without a KV cache (scoring runs with use_cache=False); generation,
which uses a cache, goes through the unmodified code. Results match eager to rounding:
check=True records each self-attention call's relative error against the unpatched code.
"""
import contextlib

import torch
import torch.nn.functional as F

QUERY_BLOCK = None        # queries per block; None picks it from BLOCK_ELEMS
BLOCK_ELEMS = 2 ** 26     # score elements per block when QUERY_BLOCK is None (~1 GB transient)

_check = None             # list of per-call records while check mode is on


class LazyDistance:
    """get_relative_distance's matrix, q_pos - kv_pos, built a slice at a time."""

    def __init__(self, q_len, kv_len, device):
        self.q_len, self.kv_len, self.device = q_len, kv_len, device

    def block(self, q0, q1, k0, k1):
        q = torch.arange(q0, q1, device=self.device)
        k = torch.arange(k0, k1, device=self.device)
        return q[:, None] - k[None, :]

    def full(self):
        return self.block(0, self.q_len, 0, self.kv_len)


class LazyBias:
    """Inkling2RelativePositionBias's output, built a slice at a time from the per-token
    relative states (batch, q_len, heads, d_rel), which the original computes first anyway."""

    def __init__(self, module, relative_states, distance, hidden_states, position_ids):
        self.module, self.relative_states, self.distance = module, relative_states, distance
        self._args = (hidden_states, position_ids)      # for check mode's reference

    def block(self, q0, q1, k0, k1):
        m = self.module
        dist = self.distance.block(q0, q1, k0, k1)
        # only distances up to the largest in this slice are gathered: the same values as
        # the original's full bank of rel_extent columns, without computing the rest
        ext = max(1, min(m.rel_extent, (q1 - 1) - k0 + 1))
        rel_logits = (self.relative_states[:, q0:q1] @ m.proj[:, :ext]).transpose(1, 2)
        index = dist.clamp(0, ext - 1).expand(*rel_logits.shape[:2], -1, -1)
        bias = rel_logits.gather(-1, index)
        return bias.masked_fill((dist < 0) | (dist >= m.rel_extent), 0.0)


def _blocked_attention(module, query, key, value, attention_mask, scaling, bias, sinks, repeat_kv):
    """eager_attention_forward's operations, a block of queries at a time."""
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    B, H, Tq, _ = query.shape
    Tk = k.shape[-2]
    assert Tq == Tk, 'blocked attention is only for scoring without a KV cache'
    window = getattr(module, 'sliding_window', None)
    span = min(Tk, window + 1) if window else Tk
    block = QUERY_BLOCK or max(16, BLOCK_ELEMS // (B * H * span))

    out = query.new_empty(B, H, Tq, v.shape[-1])
    for q0 in range(0, Tq, block):
        q1 = min(q0 + block, Tq)
        k0 = max(0, q0 - window) if window else 0     # keys before this are outside the window
        k1 = q1                                       # keys after this are in the future
        w = torch.matmul(query[:, :, q0:q1], k[:, :, k0:k1].transpose(2, 3)) * scaling
        w = w + bias.block(q0, q1, k0, k1)
        if attention_mask is not None:
            w = w + attention_mask[:, :, q0:q1, k0:k1]
        if sinks is not None:
            combined = torch.cat([w, sinks[:, :, q0:q1]], dim=-1).float()
            p = F.softmax(combined, dim=-1)[..., :-1].to(query.dtype)
        else:
            p = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, :, q0:q1] = torch.matmul(p, v[:, :, k0:k1])
        del w, p
    return out.transpose(1, 2).contiguous()


@contextlib.contextmanager
def inkling2_chunked_attention(check=False):
    """Inside the block, Inkling 2's attention runs blocked (see the module docstring).
    check=True also runs the original code on every self-attention call and records
    {'layer', 'sliding', 'rel_err'} in the list this yields."""
    global _check
    import transformers.models.inkling2.modeling_inkling2 as m

    orig_distance = m.get_relative_distance
    orig_bias = m.Inkling2RelativePositionBias.forward
    orig_eager = m.eager_attention_forward

    def get_relative_distance(inputs_embeds, cache, is_sliding):
        if cache is not None:
            return orig_distance(inputs_embeds, cache, is_sliding)
        T = inputs_embeds.shape[1]
        return LazyDistance(T, T, inputs_embeds.device)

    def bias_forward(self, hidden_states, relative_distance, position_ids):
        if not isinstance(relative_distance, LazyDistance):
            return orig_bias(self, hidden_states, relative_distance, position_ids)
        states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        states = self.norm(states, position_ids)
        return LazyBias(self, states, relative_distance, hidden_states, position_ids)

    def eager_attention_forward(module, query, key, value, attention_mask, scaling,
                                dropout=0.0, position_bias=None, sinks=None, **kwargs):
        if not isinstance(position_bias, LazyBias):
            return orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout,
                              position_bias=position_bias, sinks=sinks, **kwargs)
        out = _blocked_attention(module, query, key, value, attention_mask, scaling,
                                 position_bias, sinks, m.repeat_kv)
        if _check is not None:
            hs, pos = position_bias._args
            ref_bias = orig_bias(position_bias.module, hs, position_bias.distance.full(), pos)
            ref, _ = orig_eager(module, query, key, value, attention_mask, scaling,
                                position_bias=ref_bias, sinks=sinks)
            _check.append(dict(layer=getattr(module, 'layer_idx', None),
                               sliding=getattr(module, 'is_sliding', None),
                               rel_err=((out.float() - ref.float()).norm()
                                        / ref.float().norm()).item()))
            del ref, ref_bias
        return out, None

    m.get_relative_distance = get_relative_distance
    m.Inkling2RelativePositionBias.forward = bias_forward
    m.eager_attention_forward = eager_attention_forward
    _check = [] if check else None
    try:
        yield _check
    finally:
        m.get_relative_distance = orig_distance
        m.Inkling2RelativePositionBias.forward = orig_bias
        m.eager_attention_forward = orig_eager
        _check = None


# model class module -> patch, for ChatLM.score
PATCHES = {'transformers.models.inkling2.modeling_inkling2': inkling2_chunked_attention}


def chunked_attention_for(hf_model):
    """The blocked-attention patch for this model, or None."""
    return PATCHES.get(type(hf_model).__module__)