import gc
import os
import warnings
from contextlib import contextmanager, nullcontext
from functools import partial, cached_property

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import scipy.special as spc
import matplotlib.pyplot as plt


from vocabulary import TokenEquivalence, Vocabulary    # defined there; importable from here too


def split_block(trials, k_show=None, k_hide=None):
    """
    Context = all trials, with the last one cut; completion = the rest (no delimiter).
    Specify exactly one of:
      k_show: tokens of the last trial to present (0 = generate the whole trial)
      k_hide: tokens of the last trial to withhold
    """
    assert (k_show is None) != (k_hide is None), "specify exactly one of k_show, k_hide"
    last = trials[-1]
    n = len(last) - 1                              # real tokens, excluding the delimiter
    k = min(k_show, n - 1) if k_hide is None else n - k_hide
    assert 0 <= k < n, f"can't hide {k_hide} tokens of a {n}-token trial"
    return np.concatenate(trials[:-1] + [last[:k]]), last[k:-1]

def span_rows(a, s, e, fill):
    """Rows of per-position predictions `a` for a span s..e-1 of the sequence: those predicting
    each of its tokens, plus the one predicting the token after it."""
    if s == 0:     # raw stream without BOS: nothing predicts the very first token
        return np.concatenate([np.full((1,) + a.shape[1:], fill, a.dtype), a[:e]])
    return a[s - 1 : e]


class ChatLM:
    """
    A TransformerBridge plus the chat-template facts the ICL experiments need: prompt
    building, the thinking phase, and scoring a trial. Everything model-specific is read off
    the tokenizer and its chat template, so nothing depends on notebook globals or on one
    model family's conventions.

    end_of_reasoning: the text that ends a thought and starts the answer, e.g. '\\n</think>\\n\\n'.
        Leave as None to read it off the template (see think_tokens); pass it explicitly for
        templates where that fails.
    answer_prefix: text that opens the answer when not thinking, for templates whose
        generation prompt doesn't (Inkling 2's has none: '<|message_model|><|content_text|>').
        None: the generation cue is enough (Qwen, Gemma).
    no_think_kwargs: chat-template arguments that mean "don't think", for templates without
        an enable_thinking switch (Inkling 2: {'reasoning_effort': 'none'}); applied over
        the trial's own arguments when think=False. hf_models.chat_settings() gives both,
        and end_of_reasoning, for the models that need them.
    The model itself is still at self.bridge, for hooks, caching, etc. (with hf_models.HFBridge,
    for models transformer-lens doesn't support, there are no hooks).

    hf_model: the Hugging Face model under the bridge. Given, trials are scored with
        memory-efficient attention and the unembedding applied a chunk of positions at a
        time (see score), which long reasoning traces need; None scores through the bridge
        as before (eager attention, full-vocabulary logits for every position at once).
    Scoring settings, as attributes (not ICLModel fields, so save names don't change):
      chunked_scoring  False to score through the bridge even when hf_model is given
      score_chunk      positions unembedded at a time
      strict_kernels   allow only SDPA's flash / memory-efficient kernels, so an
                       unsupported shape raises instead of silently falling back to the
                       'math' kernel, which builds the full attention matrix again
      bypass_bridge    score with the bridge's wrappers swapped out for the HF modules they
                       wrap (see plain_hf). Some bridge components compute attention
                       themselves, eager-style, ignoring the HF attention setting.
      prefill_block    for models without SDPA (Inkling 2), positions per forward pass when
                       scoring (see hidden_states); None for a single pass
    """
    placeholder = '!'    # message content that splice_template replaces with token ids
    chunked_scoring = True
    prefill_block = 1024  # positions per forward pass when scoring a model without SDPA support
    score_chunk = 512
    strict_kernels = True
    bypass_bridge = True

    def __init__(self, bridge, prepend_bos=False, end_of_reasoning=None, hf_model=None,
                 answer_prefix=None, no_think_kwargs=None):
        self.bridge = bridge
        self.hf_model = hf_model
        self.tokenizer = bridge.tokenizer
        self.device = bridge.device
        self.bos = bridge.to_tokens('', prepend_bos=True)[0][0].item() if prepend_bos else None
        self.end_of_reasoning = end_of_reasoning
        self.answer_prefix = answer_prefix
        self.no_think_kwargs = dict(no_think_kwargs or {})

    # -- tokens ---------------------------------------------------------------

    def single_tokens(self, words, leading_space=True):
        """Ids of the words that are a single token, as ' word' if leading_space."""
        ids = []
        for w in words:
            toks = self.bridge.to_tokens(w, prepend_bos=False)[0]
            if len(toks) == 1:
                s = self.bridge.to_string(toks[0].item())
                ids.append(self.bridge.to_single_token(' ' + s if leading_space else s))
        return np.array(ids)

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    # -- chat template --------------------------------------------------------

    def chat(self, messages, think=True, tokenize=False, add_generation_prompt=True,
             **template_kwargs):
        """The chat template, by default ending in the generation cue."""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, enable_thinking=think,
            tokenize=tokenize, **template_kwargs)

    def generation_cue(self, think=True, **template_kwargs):
        """What the template appends to open the assistant turn: its header, plus whatever it
        prefills (e.g. Qwen's '<think>\\n', or an empty thought when thinking is off)."""
        msgs = [{'role': 'user', 'content': 'X'}]
        full = self.chat(msgs, think, **template_kwargs)
        bare = self.chat(msgs, think, add_generation_prompt=False, **template_kwargs)
        assert full.startswith(bare), "template renders the conversation differently with a generation prompt"
        return full[len(bare):]

    def splice_template(self, messages, contents, think, **template_kwargs):
        """
        Render `messages` with the chat template, then replace each placeholder message
        content, in order, with the corresponding token-id tensor in `contents`.
        Returns the spliced ids and the start position of each content.
        """
        ph = self._ids(self.placeholder)
        assert len(ph) == 1, f"placeholder {self.placeholder!r} isn't a single token for this tokenizer"
        ids = self.chat(messages, think, tokenize=True, return_tensors="pt",
                        **template_kwargs)['input_ids'][0]

        slots = torch.where(ids == ph[0])[0].tolist()
        assert len(slots) == len(contents), (
            f"placeholder found {len(slots)} times, expected {len(contents)}; "
            "check it isn't merging with neighbours or appearing in the system prompt")

        pieces, starts, prev, pos = [], [], 0, 0
        for slot, seg in zip(slots, contents):
            pieces.append(ids[prev:slot]); pos += slot - prev
            starts.append(pos)
            pieces.append(seg); pos += len(seg)
            prev = slot + 1
        pieces.append(ids[prev:])
        return torch.cat(pieces), starts

    def effort_prefix(self, **template_kwargs):
        """
        The text the template adds to the start of the system message in thinking mode
        (e.g. the reasoning-effort instruction plus its separator), found by comparing the
        thinking and non-thinking renders up to the system content. Empty if it adds nothing.
        """
        sentinel = 'XXSYSTEMPROMPTXX'
        msgs = [{'role': 'system', 'content': sentinel}, {'role': 'user', 'content': ''}]
        on = self.chat(msgs, think=True, **template_kwargs)
        off = self.chat(msgs, think=False, **template_kwargs)
        on, off = on[:on.index(sentinel)], off[:off.index(sentinel)]
        return on[len(os.path.commonprefix([on, off])):]

    def build_prompt(self, mode, blocks, task_toks, k_show=None, k_hide=None, system_prompt=None,
                     think=True, effort_without_thinking=False, block_text=None, **template_kwargs):
        """
        mode:
          'raw'      one continuous token stream, no chat template (self.bos, if any, first)
          'single'   the same stream, as the content of one user message
          'dialogue' one user turn per block; each non-final block is followed by an
                     assistant turn containing the true completion
        In every mode the final block stops at the cut, and its completion is returned as `target`.

        Returns input_ids, spans [(role, block, start, end)], target.
        'user' spans are each block's context; 'assistant' spans are the completions of
        non-final blocks (in 'raw'/'single' these are just stream positions, not separate turns).
        Prompt text is never in a span.

        system_prompt: {'role': 'system', 'content': ...}; in 'raw' mode, which has no system
                       message, its content starts the stream as plain text instead.
        block_text:    text to put just before each block (one string per block, '' for none):
                       at the start of the block's user turn in 'dialogue', on a line of its
                       own in the stream otherwise.
        effort_without_thinking: in non-thinking mode, add the template's reasoning-effort
                                 instruction to the system message as thinking mode would.
        """
        sys_msg = [system_prompt] if system_prompt else []
        if effort_without_thinking and not think and mode != 'raw':
            eff = self.effort_prefix(**template_kwargs)
            content = eff + system_prompt['content'] if system_prompt else eff.rstrip('\n')
            sys_msg = [{'role': 'system', 'content': content}]

        to_ids = lambda x: torch.as_tensor(task_toks[x], dtype=torch.long)
        text_ids = lambda t: torch.as_tensor(self._ids(t), dtype=torch.long)
        block_text = block_text or [''] * len(blocks)
        assert len(block_text) == len(blocks), "block_text needs one entry per block"

        # turns of (role, [(label, ids)]); unlabeled pieces are delimiters and prompt text
        turns = []
        if mode == 'raw' and system_prompt:
            turns.append(('system', [(None, text_ids(system_prompt['content'].strip() + '\n\n'))]))
        for b, trials in enumerate(blocks):
            ctx, comp = split_block(trials, k_show, k_hide)
            user = [(('user', b), to_ids(ctx))]
            if block_text[b]:
                text = block_text[b].strip()
                lead = text + '\n' if mode == 'dialogue' else '\n' + text + '\n'
                user.insert(0, (None, text_ids(lead)))
            turns.append(('user', user))
            if b < len(blocks) - 1:
                asst = [(('assistant', b), to_ids(comp))]
                if mode != 'dialogue':                      # in dialogue, the turn boundary replaces it
                    asst.append((None, to_ids(np.array([0]))))
                turns.append(('assistant', asst))
            else:
                target = to_ids(comp)

        pieces = [p for _, ps in turns for p in ps]
        lens = [len(ids) for _, ids in pieces]

        if mode == 'dialogue':
            messages = sys_msg + [{'role': role, 'content': self.placeholder} for role, _ in turns]
            input_ids, turn_starts = self.splice_template(
                messages, [torch.cat([ids for _, ids in ps]) for _, ps in turns], think,
                **template_kwargs)
            starts = [t0 + off for t0, (_, ps) in zip(turn_starts, turns)
                      for off in np.concatenate([[0], np.cumsum([len(i) for _, i in ps])[:-1]])]
        else:
            stream = torch.cat([ids for _, ids in pieces])
            if mode == 'raw':
                offset = 0 if self.bos is None else 1
                input_ids = stream if self.bos is None else torch.cat([torch.tensor([self.bos]), stream])
            elif mode == 'single':
                input_ids, (offset,) = self.splice_template(
                    sys_msg + [{'role': 'user', 'content': self.placeholder}], [stream], think, **template_kwargs)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            starts = offset + np.concatenate([[0], np.cumsum(lens)[:-1]])

        spans = [(*lab, int(s), int(s) + n) for (lab, _), s, n in zip(pieces, starts, lens) if lab]
        return input_ids, spans, target

    # -- thinking ---------------------------------------------------------------

    @cached_property
    def think_tokens(self):
        """
        (closing token id, ids that follow it, ids of the whole end-of-reasoning text).

        The end-of-reasoning text is what the non-thinking generation cue adds to the
        thinking one: Qwen prefills '<think>\\n' to think and '<think>\\n\\n</think>\\n\\n' not
        to, leaving '\\n</think>\\n\\n'; Gemma 4 opens the turn bare to think and prefills
        '<|channel>thought\\n<channel|>' not to. The closing token is its last non-whitespace
        token, which must be a single added token like '</think>' or '<channel|>'.
        """
        eor = self.end_of_reasoning
        if eor is None:
            on, off = self.generation_cue(think=True), self.generation_cue(think=False)
            if on == off or not off.startswith(on):
                raise ValueError(
                    f"can't read the end of reasoning off this template (thinking cue {on!r}, "
                    f"non-thinking cue {off!r}); pass end_of_reasoning= to ChatLM")
            eor = off[len(on):]

        ids = self._ids(eor)
        blank = [not self.tokenizer.decode([i]).strip() for i in ids]
        assert not all(blank), f"no closing token in end of reasoning {eor!r}"
        close = max(i for i, b in enumerate(blank) if not b)
        tok = self.tokenizer.convert_ids_to_tokens([ids[close]])[0]
        assert tok in self.tokenizer.get_added_vocab(), (
            f"closing marker {tok!r} isn't a single added token, so it can't be found reliably "
            "in the generated ids; pass end_of_reasoning= to ChatLM")
        start = close
        while start > 0 and blank[start - 1]:     # include the whitespace leading into it
            start -= 1
        if self.end_of_reasoning is not None:
            start = 0     # given explicitly: all of it closes a thought (Inkling 2: three tokens)

        as_ids = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        return ids[close], as_ids(ids[close + 1:]), as_ids(ids[start:])

    @torch.no_grad()
    def reason(self, input_ids, budget):
        """Let the model think (greedy) for up to `budget` tokens, then close the thought:
        keep everything up to its first closing token and what follows that, or append the
        whole end of reasoning if it never closed."""
        close_id, after_close, eot = self.think_tokens
        n = len(input_ids)
        out = self.bridge.generate(input_ids[None], max_new_tokens=budget,
                                   do_sample=False, stop_at_eos=True, verbose=False)[0]
        closes = torch.where(out[n:] == close_id)[0]
        if len(closes):
            return torch.cat([out[:n + closes[0].item() + 1], after_close])
        if len(out) - n < budget:
            warnings.warn("the model stopped before closing its reasoning (or never started); "
                          "appending the end of reasoning to whatever it produced")
            eos = getattr(self.bridge, 'eos_token_ids', None) or [self.tokenizer.eos_token_id]
            if len(out) > n and out[-1].item() in eos:
                out = out[:-1]             # the end-of-turn token it stopped on
        # append only the part of the end of reasoning it hasn't already written (Inkling 2
        # can close its thought with <|end_message|> and then stop without an answer block)
        tail, e = out[n:].tolist(), eot.tolist()
        k = next((j for j in range(min(len(tail), len(e)), 0, -1) if tail[-j:] == e[:j]), 0)
        return torch.cat([out, eot[k:]])

    def open_answer(self, input_ids, mode, think, budget=None):
        """A prompt that ends with the generation cue, carried up to where the completion
        starts. Thinking: the model's own reasoning (reason, up to `budget` tokens), or with
        budget=None an empty thought (the end of reasoning at once). Not thinking:
        answer_prefix, if any. 'raw' mode: as it is."""
        if mode == 'raw':
            return input_ids
        if think and budget is None:
            return torch.cat([input_ids, self.think_tokens[2]])
        if think:
            out = self.reason(input_ids, budget)
            torch.cuda.empty_cache()   # generation's cached blocks don't fit scoring's shapes
            return out
        if self.answer_prefix:
            return torch.cat([input_ids, torch.tensor(self._ids(self.answer_prefix),
                                                      dtype=torch.long, device=input_ids.device)])
        return input_ids

    # -- scoring --------------------------------------------------------------------

    @cached_property
    def special_ids(self):
        """Token ids that aren't text: the tokenizer's special tokens and every added token
        (for Qwen, the chat and thinking markers: <|im_end|>, <think>, ...). Check
        tokenizer.get_added_vocab() for a new model, in case it adds ordinary words."""
        tok = self.tokenizer
        return sorted(set(tok.all_special_ids) | set(tok.get_added_vocab().values()))

    def not_text(self, vocab_size):
        """special_ids plus the padding rows of the unembedding past the tokenizer's vocab."""
        return self.special_ids + list(range(len(self.tokenizer), vocab_size))

    # -- attention backend ----------------------------------------------------------

    def _attn_configs(self):
        """Every distinct HF config in the model: a multimodal model like Gemma 4 nests a
        text config inside the top one, and each attention layer reads its own config's
        _attn_implementation when it runs."""
        found = {}
        for m in self.hf_model.modules():
            c = getattr(m, 'config', None)
            if c is not None and hasattr(c, '_attn_implementation'):
                found[id(c)] = c
        return list(found.values())

    @property
    def attn_implementation(self):
        """The attention implementation(s) the HF model's configs currently name."""
        return sorted({str(c._attn_implementation) for c in self._attn_configs()})

    @contextmanager
    def attention(self, impl):
        """Run the HF model with attention implementation `impl` ('sdpa', 'eager', ...)
        inside the block, restoring what was there after. Generation and anything hooked
        through the bridge keep whatever the bridge set up (eager, so hook_pattern works)."""
        configs = self._attn_configs()
        old = [c._attn_implementation for c in configs]
        try:
            for c in configs:
                c._attn_implementation = impl
            yield
        finally:
            for c, o in zip(configs, old):
                c._attn_implementation = o

    @staticmethod
    def _wrapper_swaps(module):
        """(parent, name, wrapper, original) for every bridge component under `module`: a
        module with an `original_component`, which it stands in for. Recurses into the
        originals, where the bridge also nests its wrappers (a block's attention, ...)."""
        swaps = []
        for name, child in list(module.named_children()):
            orig = child
            while isinstance(getattr(orig, 'original_component', None), torch.nn.Module):
                orig = orig.original_component
            if orig is not child:
                swaps.append((module, name, child, orig))
            swaps += ChatLM._wrapper_swaps(orig)
        return swaps

    @contextmanager
    def plain_hf(self):
        """Inside the block, hf_model is plain transformers code: every bridge wrapper is
        replaced by the module it wraps, and put back after. No hooks fire inside."""
        swaps = self._wrapper_swaps(self.hf_model)
        try:
            for parent, name, _, orig in swaps:
                setattr(parent, name, orig)
            yield len(swaps)
        finally:
            for parent, name, wrapper, _ in reversed(swaps):
                setattr(parent, name, wrapper)

    def _kernels(self):
        if not self.strict_kernels:
            return nullcontext()
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])

    # -- scoring a sequence -----------------------------------------------------------

    @cached_property
    def _unembed(self):
        """The output layer and the final logit soft-cap, if the model has one (Gemma)."""
        cfg = self.hf_model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
        cap = getattr(text_cfg, 'final_logit_softcapping', None) or \
            getattr(cfg, 'final_logit_softcapping', None)
        return self.hf_model.get_output_embeddings(), cap

    @cached_property
    def _float_logits(self):
        """Whether the HF forward upcasts logits to float32 before the soft-cap (Inkling 2,
        whose models hold the cap as a `logit_softcap` module) rather than capping in the
        model's dtype (Gemma)."""
        return hasattr(self.hf_model, 'logit_softcap')

    def logits_from_hidden(self, h):
        """Logits for final hidden states h (already final-normed), as the HF forward
        computes them."""
        head, cap = self._unembed
        z = head(h)
        if self._float_logits:
            z = z.float()
        if cap:
            z = torch.tanh(z / cap) * cap
        return z

    @cached_property
    def sdpa_ok(self):
        """Whether the HF model runs on SDPA attention. Not all do: Inkling 2's attention adds
        per-head sinks, which transformers' SDPA function silently ignores, so forcing SDPA
        there would give wrong numbers without an error."""
        return bool(getattr(self.hf_model, '_supports_sdpa', False))

    @torch.no_grad()
    def hidden_states(self, ids):
        """
        Final hidden states (T, d_model) for a 1-d sequence of ids, in memory linear in T.

        Models with SDPA support run in one pass on SDPA, without a KV cache. Models without
        it (sdpa_ok) run their own eager attention, whose intermediates are (queries x keys):
        quadratic in T in one pass, so the sequence goes through in blocks of prefill_block
        positions instead, each attending to the earlier ones through the KV cache, and the
        intermediates are (block x T). check_blocked_scoring() compares the two ways.
        """
        with (self.plain_hf() if self.bypass_bridge else nullcontext()):
            if self.sdpa_ok:
                with self.attention('sdpa'), self._kernels():
                    out = self.hf_model.model(input_ids=ids[None], use_cache=False)
                return out.last_hidden_state[0]
            return self._hidden_states_blocked(ids, self.prefill_block)

    def _hidden_states_blocked(self, ids, block):
        """hidden_states in blocks of `block` positions through the model's KV cache (one
        pass if block is None or the sequence is no longer)."""
        if block is None or len(ids) <= block:
            return self.hf_model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        past, outs = None, []
        for s in range(0, len(ids), block):
            out = self.hf_model.model(input_ids=ids[None, s:s + block], past_key_values=past,
                                      use_cache=True)
            past = out.past_key_values
            outs.append(out.last_hidden_state[0])
        del past
        return torch.cat(outs)

    @torch.no_grad()
    def check_blocked_scoring(self, n=1500, block=500, seed=0):
        """
        Score a random n-token sequence in one pass and in blocks of `block` (how sequences
        longer than prefill_block are scored on models without SDPA), and compare the
        next-token log-probs. Run once per new model before relying on blocked scoring:
        differences should be at the level of bfloat16 rounding (around 0.01-0.1 nats at
        most, with the same top token nearly everywhere); large ones mean the model's cache
        doesn't handle multi-token blocks, and prefill_block = None is the fallback.
        """
        rng = np.random.default_rng(seed)
        ids = torch.tensor(rng.integers(1000, min(len(self.tokenizer), 100_000), n),
                           dtype=torch.long, device=self.device)
        with (self.plain_hf() if self.bypass_bridge else nullcontext()):
            lp = [torch.log_softmax(self.logits_from_hidden(self._hidden_states_blocked(ids, b)).float(), -1)
                  for b in (None, block)]
        diff = (lp[0] - lp[1]).abs().max(-1).values
        same_top = (lp[0].argmax(-1) == lp[1].argmax(-1)).float().mean().item()
        print(f'{n} tokens, blocks of {block}: max |dlogp| {diff.max().item():.3g} '
              f'(median over positions {diff.median().item():.3g}); same top token at '
              f'{100 * same_top:.1f}% of positions')
        return diff.max().item(), same_top

    @torch.no_grad()
    def score(self, ids, task_toks, equiv=None, keep=None):
        """
        Task-token log-probs and top tokens at every position of a 1-d sequence of ids:
        lp (T, n_task), top_ids (T, 2), top_lp (T, 2), as task_logprobs_gpu and
        top_tokens_gpu. The full (T, vocab) logits are never held at once.

        keep: positions (sorted, unique) at which to also return what the logits are made
        from, as a 4th value on the GPU: the final hidden states (len(keep), d_model), or
        when scoring through the bridge, the logits themselves. rep_logits turns them into
        logits; full_kl compares two sets of them over the whole vocabulary.
        """
        if not self._scores_hidden:
            return self.score_bridge(ids, task_toks, equiv, keep)
        h = self.hidden_states(ids)
        rep = None if keep is None else h[torch.as_tensor(keep, device=h.device)].clone()
        lps, tids, tlps = [], [], []
        for hc in h.split(self.score_chunk):
            z = self.logits_from_hidden(hc)
            lps.append(task_logprobs_gpu(z, task_toks, equiv, chunk=len(z)))
            i, v = top_tokens_gpu(z, self.not_text(z.shape[-1]), chunk=len(z))
            tids.append(i)
            tlps.append(v)
            del z
        del h
        out = np.concatenate(lps), np.concatenate(tids), np.concatenate(tlps)
        return out if keep is None else out + (rep,)

    @torch.no_grad()
    def score_bridge(self, ids, task_toks, equiv=None, keep=None):
        """score through the bridge, the old way: one forward with the bridge's own
        attention (eager), full logits in memory. For short checks against score."""
        logits = self.bridge(ids[None])[0]
        lp = task_logprobs_gpu(logits, task_toks, equiv)
        top_ids, top_lp = top_tokens_gpu(logits, self.not_text(logits.shape[-1]))
        rep = None if keep is None else logits[torch.as_tensor(keep, device=logits.device)].clone()
        del logits
        return (lp, top_ids, top_lp) if keep is None else (lp, top_ids, top_lp, rep)

    @property
    def _scores_hidden(self):
        """Whether score works from hidden states (the chunked path) or the bridge's logits."""
        return self.hf_model is not None and self.chunked_scoring

    def rep_logits(self, rep):
        """Logits from what score(keep=...) returned."""
        return self.logits_from_hidden(rep) if self._scores_hidden else rep

    def full_kl(self, rep_p, rep_q, task_toks=None, equiv=None):
        """KL(P ‖ Q) over the whole vocabulary, row by row, between the distributions that
        two sets of score(keep=...) representations give, a chunk of rows at a time on the
        GPU. With equiv, each task token's equivalence class counts as one outcome."""
        groups = None if equiv is None else [
            torch.as_tensor(m, dtype=torch.long, device=rep_p.device) for m in equiv.members(task_toks)]
        out = [full_kl_gpu(self.rep_logits(a), self.rep_logits(b), groups)
               for a, b in zip(rep_p.split(self.score_chunk), rep_q.split(self.score_chunk))]
        return np.concatenate(out) if out else np.zeros(0)

    @staticmethod
    def _rows_of(rep, kept, pos):
        """The rows of rep (kept at positions `kept`) for positions pos, where pos >= 0:
        (valid mask over pos, rows for the valid positions)."""
        valid = pos >= 0
        idx = torch.as_tensor(np.searchsorted(kept, pos[valid]), device=rep.device)
        return valid, rep[idx]

    @torch.no_grad()
    def run_trial(self, mode, blocks, task_toks, k_show=None, k_hide=None, system_prompt=None,
                  think=True, thinking_budget=4000, equiv=None, block_text=None,
                  return_hidden=False, **template_kwargs):
        """
        Build the prompt, think if asked (not in 'raw' mode), and score it.

        Returns logp_ctx, ans, input_ids, top:
          logp_ctx  {(role, block): (span length + 1, len(task_toks))} task-token log-probs
                    predicting each token of the span and the one after it; the final
                    block's assistant entry covers the target
          ans       the log-probs at the start of the final completion
          input_ids the full prompt including any reasoning (without the target)
          top       {'ids': {...}, 'lp': {...}}, keyed and rowed like logp_ctx: the most
                    likely token at each row, (rows, 2), over the whole vocabulary (column
                    0) and over text only (column 1: without special_ids), with its
                    full-vocabulary log-prob. -1 and nan where nothing predicts.
        With return_hidden, also a 5th value, ref: {(role, block): (valid, rows)}, keyed and
        rowed like logp_ctx, of what those rows' logits are made from (see score's keep), on
        the GPU, for run_isolated's full-vocabulary KL. valid marks the rows that something
        predicts; rows has one entry per valid row.
        """
        if not think and mode != 'raw':
            template_kwargs = {**template_kwargs, **self.no_think_kwargs}
        input_ids, spans, target = self.build_prompt(
            mode, blocks, task_toks, k_show, k_hide, system_prompt, think,
            block_text=block_text, **template_kwargs)
        input_ids, target = input_ids.to(self.device), target.to(self.device)
        input_ids = self.open_answer(input_ids, mode, think, thinking_budget)

        L = len(input_ids)
        positions = {(r, b): np.arange(s - 1, e) for r, b, s, e in spans}     # as span_rows
        positions[('assistant', len(blocks) - 1)] = np.arange(L - 1, L + len(target))
        keep = None
        if return_hidden:
            keep = np.unique(np.concatenate(list(positions.values())))
            keep = keep[keep >= 0]
        lp, top_ids, top_lp, *rep = self.score(torch.cat([input_ids, target]), task_toks, equiv, keep)

        def by_span(a, fill):
            out = {(r, b): span_rows(a, s, e, fill) for r, b, s, e in spans}
            out[('assistant', len(blocks) - 1)] = a[L - 1 : L + len(target)]
            return out

        logp_ctx = by_span(lp, np.nan)
        top = {'ids': by_span(top_ids, -1), 'lp': by_span(top_lp, np.nan)}

        # generation can leave tensors in reference cycles; collect them, then hand the
        # freed GPU blocks back so memory doesn't creep up across trials
        gc.collect()
        torch.cuda.empty_cache()
        if return_hidden:
            ref = {k: self._rows_of(rep[0], keep, p) for k, p in positions.items()}
            return logp_ctx, lp[L - 1], input_ids, top, ref
        return logp_ctx, lp[L - 1], input_ids, top


    @torch.no_grad()
    def answer(self, mode, blocks, task_toks, k_show=None, k_hide=None, system_prompt=None,
               think=True, thinking_budget=4000, equiv=None, block_text=None, reason=True,
               **template_kwargs):
        """
        The last block's completion scored as the conversation's answer, the way run_trial
        scores its final block, and nothing else. To score an earlier block b this way, pass
        blocks[:b + 1] (and block_text[:b + 1]): the conversation cut after b's user turn,
        the blocks before it laid out as in run_trial (earlier turns with their completions,
        no reasoning). Returns rows, top: the task-token log-probs predicting each token of
        the completion and the one after it, as run_trial's final assistant entry, and the
        top tokens of the same rows ({'ids', 'lp'}, as run_trial's top).
        reason: when thinking, after reasoning of the model's own (up to thinking_budget
        tokens); otherwise with the thought closed at once (an empty thought), so the two
        differ only by the reasoning. Not thinking, or in 'raw' mode, reason doesn't apply.
        """
        if not think and mode != 'raw':
            template_kwargs = {**template_kwargs, **self.no_think_kwargs}
        ids, _, target = self.build_prompt(mode, blocks, task_toks, k_show, k_hide, system_prompt,
                                           think, block_text=block_text, **template_kwargs)
        ids = self.open_answer(ids.to(self.device), mode, think,
                               thinking_budget if reason else None)
        target = target.to(self.device)
        L = len(ids)
        lp, top_ids, top_lp = self.score(torch.cat([ids, target]), task_toks, equiv)
        rows = slice(L - 1, L + len(target))
        gc.collect()
        torch.cuda.empty_cache()
        return lp[rows], {'ids': top_ids[rows], 'lp': top_lp[rows]}

    @torch.no_grad()
    def run_isolated(self, mode, blocks, task_toks, k_show=None, k_hide=None, system_prompt=None,
                     think=True, equiv=None, block_text=None, ref=None, **template_kwargs):
        """
        Every block scored without the blocks before it, and without reasoning, for comparing
        with run_trial's scores. Forward passes only: no generation. Returns logp, top, kl:
          logp  {(role, block): rows}, keyed and shaped like run_trial's logp_ctx: the task
                tokens' log-probs (normalised over the whole vocabulary) at the same tokens of
                the same trials, so kl_rows(logp_ctx[k], logp[k]) is the KL over the task
                alphabet
          top   {'ids': {...}, 'lp': {...}}, as run_trial's top, for the same rows
          kl    with ref (run_trial's 5th value, from return_hidden=True), the KL from
                run_trial's distribution to this one over the whole vocabulary, row by row
                (see full_kl): {(role, block): (rows,)}, nan where nothing predicts; else None

        Block b is scored at the start of a prompt of its own, laid out as a first block:
        the same system prompt, and block_text[b] before it. So pass each block's text as it
        would be for a first block (ICLModel's _prompt gives it: no transition from a previous
        block). Rows only depend on what comes before them, so a non-final block is scored on
        a prompt cut off right after its completion, which is laid out as it is in context
        (in 'dialogue', an earlier assistant turn). The final block's completion is scored
        straight after the generation cue: when thinking, the prompt is rendered for thinking
        as in run_trial, and the thought is closed at once with the end of reasoning (an empty
        thought), so against run_trial the comparison measures the earlier blocks and the
        reasoning trace together; otherwise after answer_prefix, as in run_trial.
        """
        block_text = block_text or [''] * len(blocks)
        assert len(block_text) == len(blocks), "block_text needs one entry per block"
        prompt_kwargs = (dict(template_kwargs, **self.no_think_kwargs)
                         if not think and mode != 'raw' else template_kwargs)
        logp, top, kl = {}, {'ids': {}, 'lp': {}}, (None if ref is None else {})

        def take(seq, positions):
            """Score seq and fill in logp (and kl) for {key: positions}."""
            keep = None
            if ref is not None:
                keep = np.unique(np.concatenate(list(positions.values())))
                keep = keep[keep >= 0]
            lp, top_ids, top_lp, *rep = self.score(seq, task_toks, equiv, keep)
            for key, pos in positions.items():
                ok = pos >= 0                                   # nothing predicts pos -1
                for out, a, fill in [(logp, lp, np.nan), (top['ids'], top_ids, -1),
                                     (top['lp'], top_lp, np.nan)]:
                    out[key] = np.full((len(pos),) + a.shape[1:], fill, a.dtype)
                    out[key][ok] = a[pos[ok]]
                if ref is not None:
                    vp, hp = ref[key]
                    vq, hq = self._rows_of(rep[0], keep, pos)
                    assert len(vp) == len(pos), f"{key}: {len(vp)} rows in context, {len(pos)} alone"
                    both = vp & vq
                    kl[key] = np.full(len(pos), np.nan)
                    kl[key][both] = self.full_kl(hp[torch.as_tensor(both[vp], device=hp.device)],
                                                 hq[torch.as_tensor(both[vq], device=hq.device)],
                                                 task_toks, equiv)

        for b in range(len(blocks) - 1):
            # the block followed by a final block that is cut off again: only so that the
            # block is laid out as a non-final one, with its completion in context
            ids, spans, _ = self.build_prompt(
                mode, [blocks[b], blocks[b]], task_toks, k_show, k_hide, system_prompt, think,
                block_text=[block_text[b], ''], **prompt_kwargs)
            mine = {(r, b): np.arange(s - 1, e) for r, blk, s, e in spans if blk == 0}
            end = max(p[-1] for p in mine.values()) + 1
            take(ids[:end].to(self.device), mine)

        last = len(blocks) - 1
        ids, spans, target = self.build_prompt(
            mode, blocks[-1:], task_toks, k_show, k_hide, system_prompt, think,
            block_text=block_text[-1:], **prompt_kwargs)
        ids, target = ids.to(self.device), target.to(self.device)
        ids = self.open_answer(ids, mode, think)                    # an empty thought
        L = len(ids)
        positions = {(r, last): np.arange(s - 1, e) for r, _, s, e in spans}
        positions[('assistant', last)] = np.arange(L - 1, L + len(target))
        take(torch.cat([ids, target]), positions)
        return logp, top, kl

def test_token_logits(logs, spans, blocks, Y, task_toks, k_show=None, k_hide=None, ans=None):
    """
    For every trial, the model's logits at the position predicting its test token (the first
    hidden token, by the same k_show / k_hide rule as split_block), the generative distribution
    of that token over the full vocabulary, and the trial's block.

    logs: (T, vocab) logits for the full input
    Y:    (len(X), V) generative distribution over task symbols
    ans:  optional (vocab,) logits from the start of the final assistant turn; appended as an
          extra last row, paired with the same target as the final trial

    Returns logits (n_trials[+1], vocab), target (same shape), block (n_trials[+1],)
    """
    assert (k_show is None) != (k_hide is None), "specify exactly one of k_show, k_hide"
    user_start = {b: s for r, b, s, e in spans if r == 'user'}

    rows, xs, blk = [], [], []
    t0 = 0
    for b, trials in enumerate(blocks):
        pos = t0
        for t in trials:
            n = len(t) - 1                                       # real tokens, excluding delimiter
            k = min(k_show, n - 1) if k_hide is None else n - k_hide
            xs.append(pos + k)
            rows.append(user_start[b] - 1 + (pos + k - t0))      # row predicting that X position
            blk.append(b)
            pos += len(t)
        t0 = pos
    rows, xs, blk = np.array(rows), np.array(xs), np.array(blk)

    logits = np.full((len(rows), logs.shape[1]), np.nan, dtype=logs.dtype)
    ok = rows >= 0                                               # raw, no BOS: nothing predicts X[0]
    logits[ok] = logs[rows[ok]]

    target = np.zeros((len(rows), logs.shape[1]))
    target[:, task_toks] = Y[xs]

    if ans is not None:
        logits = np.vstack([logits, ans[None]])
        target = np.vstack([target, target[-1:]])
        blk = np.append(blk, blk[-1])
    return logits, target, blk

def xent(lp, y, renorm=True):
    """Expected log-prob under Y of the task-token log-probs lp. renorm: after renormalising
    them over the task alphabet; otherwise as they are, normalised over the whole
    vocabulary, so probability the model puts on other tokens counts against it. The two
    differ by log(task mass): xent(lp, y) == xent(lp, y, False) - logsumexp(lp)."""
    if renorm:
        lp = lp - spc.logsumexp(lp, -1, keepdims=True)
    with np.errstate(invalid='ignore'):
        return np.where(y > 0, lp * y, 0).sum(-1)       # 0 * -inf counts as 0

def kl_rows(lp, lq, renorm=True):
    """KL(P ‖ Q) row by row, from task-token log-probs lp (P) and lq (Q) of the same shape,
    e.g. a span of run_trial's logp_ctx and the same span of run_isolated's. renorm: between
    the distributions renormalised over the task alphabet; otherwise over the task tokens plus
    one outcome for the rest of the vocabulary, so a shift of probability onto or off the task
    tokens counts too. nan where either row is nan."""
    def dist(l):
        l = np.asarray(l, float)
        lse = spc.logsumexp(l, -1, keepdims=True)
        if renorm:
            return l - lse
        with np.errstate(divide='ignore'):                   # all mass on the task: log 0
            return np.concatenate([l, np.log(-np.expm1(np.minimum(lse, 0)))], -1)
    lp, lq = dist(lp), dist(lq)
    with np.errstate(invalid='ignore'):
        terms = np.exp(lp) * (lp - lq)
    return np.where(lp == -np.inf, 0, terms).sum(-1)       # 0 log 0 counts as 0; nan stays nan

def mean_sem(a):
    a = np.asarray(a, dtype=float)
    return np.nanmean(a, 0), np.nanstd(a, 0)


def spread(a, err='std'):
    """
    Centre and error bars of `a` (samples x points), over the samples, by `err`:
        'std'       the mean, +/- the standard deviation
        'iqr'       the median, from the 25th to the 75th percentile
        (lo, hi)    the median, from the lo to the hi quantile, e.g. (0.1, 0.9)
        None        the mean, and no bars
    Returns the centre (points,) and yerr (2, points), the distances below and above it,
    as errorbar takes them (None for no bars).
    """
    a = np.asarray(a, float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-nan columns: gaps
        if err is None or err == 'std':
            mu = np.nanmean(a, 0)
            if err is None:
                return mu, None
            sd = np.nanstd(a, 0)
            return mu, np.stack([sd, sd])
        if err == 'iqr':
            err = (0.25, 0.75)
        if not (isinstance(err, (tuple, list)) and len(err) == 2):
            raise ValueError(f"err is 'std', 'iqr', a pair of quantiles or None, not {err!r}")
        q = np.nanquantile(a, [err[0], 0.5, err[1]], axis=0)
    return q[1], np.stack([q[1] - q[0], q[2] - q[1]])


def draw_traces(ax, x, rows, color, alpha=True):
    """Each row of `rows` (samples x points) as a thin translucent line over x, joining
    its scored points across gaps. alpha=True is 0.15; a number sets it."""
    alpha = 0.15 if alpha is True else alpha
    for r in np.atleast_2d(np.asarray(rows, float)):
        ok = np.isfinite(r)
        ax.plot(np.asarray(x)[ok], r[ok], color=color, lw=0.6, alpha=alpha, zorder=1)
# Reference predictors: what a simple strategy would predict for each token.
# Each takes (data, i) and returns P, shape (len(X), V): row t is the predicted
# distribution of token t of sample i (aligned with data['Y'][i]). Fix any other
# arguments with functools.partial and pass them to plot_icl as refs.

def block_code(data, i):
    """code[b, r]: the token that the grammar's token r becomes in block b of
    sample i (0, the delimiter, stays 0). Data sampled before GrammarTask kept
    'code' is assumed to have disjoint alphabets (share_toks=False), and checked."""
    if 'code' in data:
        return data['code'][i]
    X, B, nblk, V = data['X'][i], data['B'][i], len(data['blocks'][i]), data['V']
    nt = (V - 1) // nblk + 1
    code = np.zeros((nblk, nt), int)
    code[:, 1:] = np.arange(1, nt) + (nt - 1) * np.arange(nblk)[:, None]
    for b in range(nblk):
        x = X[B == b]
        if not np.isin(x, code[b]).all():
            raise ValueError("can't recover the block alphabets (shared tokens?): resample the data")
    return code

def token_kinds(grammar):
    """The kind of each of an automaton's tokens: 0 for the delimiter, then one
    number per kind (action, node, ...), as in the automaton's KINDS."""
    kind = np.zeros(len(grammar.vocab), int)
    for k, table in enumerate(grammar._token_of.values()):
        kind[list(table.values())] = k + 1
    return kind

def true_dist(data, i):
    """The true distribution Y, so the score is -H(Y)."""
    return data['Y'][i]

def uniform_block(data, i):
    """Uniform over the alphabet of the token's block, the delimiter included."""
    B, V, code = data['B'][i], data['V'], block_code(data, i)
    P = np.zeros((len(B), V))
    for b in np.unique(B):
        P[np.ix_(B == b, code[b])] = 1.0 / code.shape[1]
    return P

def uniform_kind(data, i, grammar):
    """Uniform over the tokens of the target's kind (action, node, ... for an
    automata.Automaton; the delimiter is a kind of its own) in its block's
    alphabet. An oracle on the kind, so it knows the grammar's syntax."""
    X, B, V, code = data['X'][i], data['B'][i], data['V'], block_code(data, i)
    kind = token_kinds(grammar)
    P = np.zeros((len(X), V))
    for b in np.unique(B):
        enc = np.full(V, -1)
        enc[code[b]] = kind                                  # kind of each token in block b
        for k in np.unique(kind):
            members = code[b][kind == k]
            P[np.ix_((B == b) & (enc[X] == k), members)] = 1.0 / len(members)
    return P

def ngram(data, i, order=1, alpha=1.0):
    """The empirical transitions seen so far: each token predicted from the
    frequencies of what followed the same `order` previous tokens earlier in
    its block, plus `alpha` pseudocounts on each token of the block's
    alphabet. order=1 is first-order (bigram) statistics; order=0 the token
    frequencies alone. The context before a block's first token is padded with
    the delimiter, and contexts run across trial boundaries, as the stream
    does. With alpha=0 an unseen context predicts uniformly, but an unseen
    transition scores -inf."""
    X, B, V, code = data['X'][i], data['B'][i], data['V'], block_code(data, i)
    P = np.zeros((len(X), V))
    for b in np.unique(B):
        prior = np.zeros(V)
        prior[code[b]] = 1.0
        counts = {}
        ctx = (0,) * order
        for t in np.where(B == b)[0]:
            row = counts.get(ctx, 0.0) + alpha * prior
            P[t] = row / row.sum() if row.sum() > 0 else prior / prior.sum()
            counts.setdefault(ctx, np.zeros(V))[X[t]] += 1
            ctx = (ctx + (X[t],))[1:]
    return P

# Kernel-model references, for automaton walks (node, action, node, ...). Each node that
# follows an action in a block is predicted from the block's earlier transitions
# u -a-> v, as a regression from the indicators [eye(N)[u], eye(M)[a]] to eye(N)[v]; any
# other token gets `fallback` (uniform_block by default). Only the nodes of the token's
# block are candidates.

def _kinds(data, i):
    """Per block, {symbol: 'node' | 'action' | None} for the block's alphabet."""
    code, kinds = block_code(data, i), data.get('kinds')
    if kinds is None:
        raise ValueError("these references need data['kinds'] (runs saved since it was added)")
    return [{int(s): kinds[r] for r, s in enumerate(code[b])} for b in range(len(code))]


def _relational_ref(data, i, predict, eps=0.01, fallback=None):
    """P with each node after an action given by predict(seen, u, a, nodes): seen is the
    block's earlier transitions [(u, a, v)] as symbols (repeats included), nodes the
    block's node symbols; it returns weights over nodes (>= 0, not all 0). Mixed with
    eps of the uniform over nodes, so a miss scores a finite log-likelihood."""
    X, B = data['X'][i], data['B'][i]
    P = np.array((fallback or uniform_block)(data, i), float)
    for b, kind in enumerate(_kinds(data, i)):
        nodes = np.array([s for s, k in kind.items() if k == 'node'])
        ts = np.flatnonzero(B == b)
        seen = []
        for t in ts[2:]:
            u, a, v = (int(x) for x in X[t - 2:t + 1])
            if not (kind.get(u) == 'node' and kind.get(a) == 'action' and kind.get(v) == 'node'):
                continue
            w = np.asarray(predict(seen, u, a, nodes), float)
            p = w / w.sum() if w.sum() > 0 else np.full(len(nodes), 1 / len(nodes))
            P[t] = 0
            P[t, nodes] = (1 - eps) * p + eps / len(nodes)
            seen.append((u, a, v))
    return P


def _choose(scores, beta):
    """Scores over the candidates -> weights: uniform on the best (beta=None), or
    softmax(beta * scores)."""
    s = np.asarray(scores, float)
    if beta is None:
        return np.isclose(s, s.max(), atol=1e-9).astype(float)
    return np.exp(beta * (s - s.max()))


def kernel_ref(data, i, kappa=(0, 1, 1, 2), ridge=0.0, counts=False, beta=None, w_self=0.0,
               eps=0.01, fallback=None):
    """
    Kernel (ridge) regression from (u, a) to v on the block's earlier transitions, the
    kernel depending only on which labels two inputs share:
        kappa = (k0, ku, ka, k2): k0 sharing neither, ku the source only, ka the action
        only, k2 both. (0, 1, 1, 2) is the linear kernel on [eye(N)[u], eye(M)[a]]
        (min-norm least squares: additive, parallelogram completion); an RBF on those
        indicators is rbf_kappa(gamma). Positive semi-definite when every eigenvalue
        on the full grid of N nodes x M actions is >= 0: k2-ku-ka+k0,
        k2+(M-1)ku-ka-(M-1)k0, k2-ku+(N-1)ka-(N-1)k0, k2+(M-1)ku+(N-1)ka+(N-1)(M-1)k0.
    ridge: added to the training kernel's diagonal (0: interpolate, via pinv).
    counts: train on every earlier transition, repeats included, rather than each
    distinct one once (the analysis used distinct ones).
    The prediction y_hat (one score per node), plus w_self on the source itself, goes
    to the best node(s) (beta=None) or softmax(beta * scores). A pair (u, a) seen
    before is interpolated back to its target unless ridge > 0.
    """
    k0, ku, ka, k2 = kappa

    def predict(seen, u, a, nodes):
        tr = seen if counts else list(dict.fromkeys(seen))
        s = np.zeros(len(nodes))
        if tr:
            U = np.array([x[0] for x in tr]); A = np.array([x[1] for x in tr])
            Y = (np.array([x[2] for x in tr])[:, None] == nodes[None]).astype(float)
            su, sa = U[:, None] == U[None], A[:, None] == A[None]
            K = k0 + (ku - k0) * su + (ka - k0) * sa + (k2 - ku - ka + k0) * (su & sa)
            ku_, ka_ = U == u, A == a
            k = k0 + (ku - k0) * ku_ + (ka - k0) * ka_ + (k2 - ku - ka + k0) * (ku_ & ka_)
            s = k @ np.linalg.pinv(K + ridge * np.eye(len(K))) @ Y
        s = s + w_self * (nodes == u)
        return _choose(s, beta)

    return _relational_ref(data, i, predict, eps, fallback)


def rbf_kappa(gamma):
    """kernel_ref's kappa for an RBF exp(-gamma |x - x'|^2) on the concatenated
    indicators: inputs differing in one label are at squared distance 2."""
    r = np.exp(-2 * gamma)
    return (r * r, r, r, 1.0)


def cue_ref(data, i, weights=(0, 1, 0), w_self=0.0, counts=False, beta=None, eps=0.01,
            fallback=None):
    """
    The cue heuristics: for an unseen pair (u, a), each node v scores
        w_src * #(u -b-> v, b != a) + w_act * #(x -a-> v, x != u) + w_nei * #(x -b-> v,
        x != u, b != a) + w_self [v == u]
    over the block's earlier transitions (each distinct one once, or with counts every
    repeat), with weights = (w_src, w_act, w_nei) -- the per-training-point weights a
    kernel puts on each group when the rest of the (node x action) grid is seen, e.g.
    (3, 1, -1) for the linear kernel. The scores go to the best node(s) (beta=None)
    or softmax(beta * scores). A pair (u, a) seen before predicts what followed it.
    """
    w_src, w_act, w_nei = weights

    def predict(seen, u, a, nodes):
        known = [v for uu, aa, v in seen if (uu, aa) == (u, a)]
        if known:
            return (np.array(known)[:, None] == nodes[None]).sum(0)
        tr = seen if counts else list(dict.fromkeys(seen))
        s = np.zeros(len(nodes))
        for uu, aa, v in tr:
            w = w_src if uu == u else w_act if aa == a else w_nei
            s += w * (nodes == v)
        s += w_self * (nodes == u)
        return _choose(s, beta)

    return _relational_ref(data, i, predict, eps, fallback)


def copy_or_sink(data, i, eps=0.01, fallback=None):
    """
    The rule that matched gemma's choices on record512 (71 of 80 tests): if the source
    u has started a transition before, predict what followed it, by how often, whatever
    the action; otherwise uniform over the 'sink' nodes (seen only as targets, never
    as sources), u excluded; failing that over every node seen as a target but u; and
    failing that over all the block's nodes.
    """
    def predict(seen, u, a, nodes):
        after = [v for uu, _, v in seen if uu == u]
        if after:
            return (np.array(after)[:, None] == nodes[None]).sum(0)
        srcs, tgts = {x[0] for x in seen}, {x[2] for x in seen}
        for cand in ((tgts - srcs) - {u}, tgts - {u}):
            if cand:
                return np.isin(nodes, list(cand)).astype(float)
        return np.ones(len(nodes))

    return _relational_ref(data, i, predict, eps, fallback)


KERNEL_REFS = {
    'linear kernel': partial(kernel_ref, kappa=(0, 1, 1, 2)),
    'RBF, gamma=1': partial(kernel_ref, kappa=rbf_kappa(1.0)),
    'copy source': partial(cue_ref, weights=(1, 0, 0)),
    'action majority': partial(cue_ref, weights=(0, 1, 0)),
    'other-action majority': partial(cue_ref, weights=(0, 0, 1)),
    'target frequency': partial(cue_ref, weights=(1, 1, 1)),
    'anti-frequency': partial(cue_ref, weights=(-1, -1, -1)),
    'avoid copy, anti-frequency': partial(cue_ref, weights=(-0.92, -0.33, -0.23)),
    'copy or sink': copy_or_sink,
}
"""The references discussed, by name: refs=KERNEL_REFS, or pick some, e.g.
{k: KERNEL_REFS[k] for k in ['linear kernel', 'action majority']}. They are rules
(the best node(s), with eps = 0.01 elsewhere); partial(kernel_ref, beta=...) or
partial(cue_ref, beta=...) gives a graded version."""

REF_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']

def ylabel(perplexity=False, kl=False, renorm=True):
    """The y-axis label for a score from plot_icl (or icl_sweeps), given how it's shown."""
    if kl:
        out = 'perplexity / true perplexity (exp KL)' if perplexity else '−KL(Y ‖ model)'
    else:
        out = 'perplexity (exp cross-entropy)' if perplexity else 'log-likelihood (−cross-entropy)'
    return out if renorm else out + ', whole vocabulary'

def plot_icl(ax, logp_ctx, data, n_trl, k_hide=None, show_assistant=False, title='', cmap='viridis',
             refs=(), perplexity=False, kl=False, err='std', traces=False, renorm=True, iso=None,
             assistant=None):
    """
    n_trl:  number of trials (comma-separated segments) per block to show, from the block's start
    k_hide: if given, show only the last k_hide tokens of each trial; otherwise every token
    show_assistant: also plot, beside the trials, each block's prediction of its first
                    completion token from the start of the assistant turn (one star per block).
                    Only meaningful for the 'dialogue' and 'single' formats.
    refs:   {label: predictor}, reference predictors (see true_dist, uniform_block,
            uniform_kind, ngram, and for relations kernel_ref, cue_ref, copy_or_sink and
            KERNEL_REFS) scored at the same tokens as the model, and drawn with
            hollow markers and dashed lines in each block's colour. E.g.
                refs={'true': true_dist, 'uniform': uniform_block,
                      'kind': partial(uniform_kind, grammar=task.grammar),
                      'bigram': partial(ngram, order=1, alpha=0.5)}
    perplexity: plot exp(cross-entropy) of each token, taken before averaging over samples,
                instead of log-likelihood; the references likewise.
    kl:     plot -KL(Y || model), the log-likelihood less the best achievable (-H(Y)), so that
            0 is the true distribution; the references likewise. The same as the
            log-likelihood wherever Y is deterministic. With perplexity, exp(KL): the
            model's perplexity as a multiple of the true distribution's.
    err:    the error bars, over samples (see spread): 'std' (mean +/- std, the default),
            'iqr' (median, 25th to 75th percentile), a pair of quantiles around the
            median, or None. The references' lines follow the same centre.
    traces: also draw every sample's curve, thin and translucent behind the summary
            (True, or the opacity to use).
    renorm: score the model's probabilities renormalised over the task alphabet (the
            default); False keeps them normalised over the whole vocabulary, so
            probability on other tokens (e.g. <|im_end|>) counts against it. The
            references are distributions over the task alphabet either way.
    iso:    plot KL(with ‖ without) at each token instead: how much the earlier blocks change
            the model's prediction, and when thinking, the reasoning trace too, at the final
            block's completion (see ChatLM.run_isolated). One dict per sample, either
              metrics['kl_iso']   the KL over the whole vocabulary, as stored (renorm
                                  doesn't apply), or
              metrics['logp_iso'] the same blocks' task-token log-probs, for the KL over the
                                  task alphabet (renorm=False: the task tokens and the rest
                                  of the vocabulary as one more outcome; see kl_rows)
            Block 0 has no earlier blocks, so its curve is a check that the two runs line up:
            0 up to numerical noise. Doesn't combine with refs, kl or perplexity.
    assistant: where the assistant turn's rows come from, one dict per sample with an
            ('assistant', b) entry for every block: e.g. ICLModel's metrics['logp_think'] (each
            block answered after reasoning of its own) or metrics['logp_nothink'] (with the
            thought closed at once). Default: logp_ctx's (earlier blocks' completions as
            earlier turns, the final block's after reasoning when thinking). With iso, only
            logp_iso: kl_iso is stored for logp_ctx's rows.
    """
    if iso is not None and (refs or kl or perplexity):
        raise ValueError("iso plots KL(with ‖ without earlier blocks), which takes no refs, kl or perplexity")
    if assistant is not None and iso is not None and np.ndim(iso[0][('user', 0)]) == 1:
        raise ValueError("kl_iso is stored for logp_ctx's assistant rows; with assistant=..., pass "
                         "iso=logp_iso")
    refs = dict(refs)
    f = (lambda v: np.exp(-v)) if perplexity else (lambda v: v)
    stored_kl = iso is not None and np.ndim(iso[0][('user', 0)]) == 1    # kl_iso, not logp_iso

    nblk = len(data['blocks'][0])
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.85, nblk))
    W = k_hide if k_hide else max(len(t) for blks in data['blocks'] for blk in blks for t in blk)

    def to_grid(vals, trials):
        """Per-token values of one block -> (n_trl, W), aligned as the trials are shown."""
        grid = np.full((n_trl, W), np.nan)
        pos = 0
        for j, t in enumerate(trials):
            if j < n_trl:
                n = len(t) - 1
                if k_hide:
                    seg = vals[pos + max(n - k_hide, 0) : pos + n]
                    grid[j, W - len(seg):] = seg
                else:
                    seg = vals[pos:pos + len(t)]
                    grid[j, :len(seg)] = seg
            pos += len(t)
        return grid

    curves = {k: [[] for _ in range(nblk)] for k in ['model', *refs]}
    stars = [[] for _ in range(nblk)]
    for i in range(len(logp_ctx)):
        Y, B, blocks = data['Y'][i], data['B'][i], data['blocks'][i]
        best = spc.xlogy(Y, Y).sum(-1) if kl else np.zeros(len(Y))  # -H(Y), what kl subtracts
        ref = {}                                                    # per-position reference scores
        for k, pred in refs.items():
            P = pred(data, i)
            ref[k] = spc.xlogy(Y, P / P.sum(-1, keepdims=True)).sum(-1) - best

        for b in range(nblk):
            t0 = np.argmax(B == b)
            vals = np.full(sum(len(t) for t in blocks[b]), np.nan)

            ctx = logp_ctx[i][('user', b)]
            asst = (logp_ctx if assistant is None else assistant)[i][('assistant', b)]
            rest, s = asst[1:-1], len(ctx)                          # completion tokens after the first
            if iso is None:                                         # against the true distribution
                score = lambda lp, y: xent(lp, y, renorm)
                y_ctx, y_rest, y_star = Y[t0:t0 + s], Y[t0 + s:t0 + s + len(rest)], Y[t0 + s - 1]
            else:                                                   # against the same rows, block alone
                score = (lambda lp, kl: kl) if stored_kl else (lambda lp, lq: kl_rows(lp, lq, renorm))
                q_asst = iso[i][('assistant', b)]
                y_ctx, y_rest, y_star = iso[i][('user', b)], q_asst[1:-1], q_asst[0]
            vals[:s] = score(ctx, y_ctx)
            if len(rest):
                vals[s:s + len(rest)] = score(rest, y_rest)
            vals -= best[t0:t0 + len(vals)]
            stars[b].append(f(score(asst[0], y_star)                # first completion token, assistant turn
                              - best[t0 + s - 1]))

            curves['model'][b].append(to_grid(f(vals), blocks[b]))
            for k, r in ref.items():                                # only where the model is scored
                r = r[t0:t0 + len(vals)].copy()
                r[np.isnan(vals)] = np.nan
                curves[k][b].append(to_grid(f(r), blocks[b]))

    x = np.arange(n_trl * (W + 1))
    def flat(g):
        g = np.asarray(g)
        g = np.concatenate([g, np.full(g.shape[:2] + (1,), np.nan)], axis=2)  # gap between trials
        return g.reshape(len(g), -1)

    style = dict(ls='--', lw=0.8, ms=5, mfc='none')
    for b in range(nblk):
        F = flat(curves['model'][b])
        if traces:
            draw_traces(ax, x, F, colors[b], traces)
        mu, e = spread(F, err)
        ax.errorbar(x, mu, e, color=colors[b], marker='o' if k_hide else None, ms=4,
                    label=f'block {b}')
        for m, k in enumerate(refs):
            mu = spread(flat(curves[k][b]), err)[0]
            mu[~np.isfinite(mu)] = np.nan                           # -inf (unseen n-gram) as a gap too
            ax.plot(x, mu, color=colors[b], alpha=0.8,
                    marker=REF_MARKERS[m % len(REF_MARKERS)], **style)
    for m, k in enumerate(refs):                                    # one legend entry per reference
        ax.plot([], [], color='0.4', marker=REF_MARKERS[m % len(REF_MARKERS)], label=k, **style)

    ticks = list(np.arange(n_trl) * (W + 1) + (W - 1) / 2)
    labels = [str(j + 1) for j in range(n_trl)]

    if show_assistant:
        x_side = n_trl * (W + 1) + 1
        ax.axvline(x_side - 1, color='0.8', lw=1)
        offsets = np.linspace(-0.4, 0.4, nblk) if nblk > 1 else [0.0]
        for b in range(nblk):
            s = np.asarray(stars[b], float)
            if traces:                                              # one dot per sample
                ax.plot(np.full(len(s), x_side + offsets[b]), s, '.', color=colors[b],
                        alpha=0.15 if traces is True else traces, zorder=1)
            mu, e = spread(s[:, None], err)
            ax.errorbar([x_side + offsets[b]], mu, e, color=colors[b], marker='*', ms=12,
                        ls='none', label='assistant turn' if b == nblk - 1 else None)
        ticks.append(x_side)
        labels.append('asst')

    # ax.set_xticks(ticks, labels)
    ax.set_title(title)
    ax.set_xlabel('trial')
    ax.set_ylabel(ylabel(perplexity, kl, renorm) if iso is None else
                  'KL(with ‖ without earlier blocks)' + (', whole vocabulary' if stored_kl else
                                                         ', task alphabet' if renorm else
                                                         ', task tokens + rest'))
    ax.legend(loc='upper right' if perplexity or iso is not None else 'lower right', fontsize='small')


"""Token equivalence: collapse nuisance variants (e.g. 'nature' vs ' nature') before scoring"""

def task_logprobs_gpu(logits, task_toks, equiv=None, chunk=512):
    """task_logprobs for a logits tensor, computed where it lives (the GPU) a chunk of
    positions at a time. Only the (T, len(task_toks)) result comes back to the CPU, never
    the full-vocabulary logits, which are several GB in float32 for a long prompt."""
    if equiv is None:
        cols = torch.as_tensor(np.asarray(task_toks), dtype=torch.long, device=logits.device)
    else:
        groups = [torch.as_tensor(m, dtype=torch.long, device=logits.device)
                  for m in equiv.members(task_toks)]
    out = []
    for c in logits.split(chunk):
        c = c.float()
        lse = torch.logsumexp(c, dim=-1, keepdim=True)
        if equiv is None:
            out.append(c[:, cols] - lse)
        else:
            out.append(torch.stack([torch.logsumexp(c[:, g], dim=-1) for g in groups], dim=-1) - lse)
    return torch.cat(out).cpu().numpy()



def full_kl_gpu(zp, zq, groups=None):
    """KL(P ‖ Q) per row between the distributions of two logits tensors (rows, vocab), on
    their device; returned on the CPU. groups: index tensors of tokens to count as one
    outcome each (task tokens' equivalence classes), or None."""
    lp, lq = torch.log_softmax(zp.float(), -1), torch.log_softmax(zq.float(), -1)
    terms = torch.where(lp > -torch.inf, lp.exp() * (lp - lq), 0.0)        # 0 log 0 = 0
    kl = terms.sum(-1)
    if groups:
        members = torch.cat(groups)
        kl = kl - terms[:, members].sum(-1)
        cp = torch.stack([torch.logsumexp(lp[:, g], -1) for g in groups], -1)
        cq = torch.stack([torch.logsumexp(lq[:, g], -1) for g in groups], -1)
        kl = kl + torch.where(cp > -torch.inf, cp.exp() * (cp - cq), 0.0).sum(-1)
    return kl.cpu().numpy()

def top_tokens_gpu(logits, exclude=(), chunk=512):
    """The most likely token at each position, over the whole vocabulary and over all but
    the ids in `exclude`, computed on the GPU a chunk at a time like task_logprobs_gpu.
    Returns ids (T, 2) int32 and their full-vocabulary log-probs (T, 2) float32."""
    ex = torch.as_tensor(list(exclude), dtype=torch.long, device=logits.device)
    ids, lps = [], []
    for c in logits.split(chunk):
        c = c.float()
        c = c - torch.logsumexp(c, dim=-1, keepdim=True)
        v0, i0 = c.max(dim=-1)
        c[:, ex] = -torch.inf
        v1, i1 = c.max(dim=-1)
        ids.append(torch.stack([i0, i1], dim=-1))
        lps.append(torch.stack([v0, v1], dim=-1))
    return (torch.cat(ids).cpu().numpy().astype(np.int32),
            torch.cat(lps).cpu().numpy().astype(np.float32))


def task_logprobs(logs, task_toks, equiv=None):
    """Full-vocab log-probs of the task tokens, with each token's equivalence class
    collapsed into it if equiv is given.

    Only the model side needs collapsing: the generative distribution Y puts all its mass
    on the task tokens themselves, one per class (members() checks this), so collapsing Y
    onto the classes leaves it unchanged. xent's renormalisation over the task alphabet
    then becomes a renormalisation over the task classes."""
    if equiv is None:
        return logs[:, task_toks] - spc.logsumexp(logs, axis=-1, keepdims=True)
    return equiv.logprobs(logs, task_toks)