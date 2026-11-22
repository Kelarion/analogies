"""
Loading Hugging Face models for the experiments, whatever their architecture:

    hf_model = load_model(name)                  # the auto-class its config calls for
    bridge = make_bridge(name, hf_model, tok)    # transformer-lens' bridge, or HFBridge if
                                                 # transformer-lens doesn't know the model
    ChatLM(bridge, **chat_settings(hf_model.config), ...)

Written for Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, model_type inkling2_loop, needs
transformers >= 5.24), which differs from the Qwen / Gemma models in all three:
  - it registers as an image-and-text model, not under AutoModelForCausalLM;
  - its attention is eager only (sinks and a relative position bias), so no 'sdpa';
  - transformer-lens has no adapter for it;
  - its chat template has no generation prompt and no thinking switch (see CHAT_SETTINGS).
"""
import warnings

import torch


# -- loading ---------------------------------------------------------------------------------

def auto_class(config):
    """The transformers auto-class that accepts this config: AutoModelForCausalLM for text
    models, AutoModelForImageTextToText for multimodal ones whose text model isn't registered
    as a causal LM on its own (Inkling 2: only its text part, inkling2_loop_text, is)."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    for auto in (AutoModelForCausalLM, AutoModelForImageTextToText):
        if type(config) in auto._model_mapping:
            return auto
    raise ValueError(f"no auto-class loads {type(config).__name__} (model_type "
                     f"{config.model_type!r}); is transformers recent enough for this model?")


def supports_sdpa(config):
    """Whether the model class for this config runs on SDPA attention. Some don't (Inkling 2:
    its attention adds per-head sinks, which transformers' SDPA function silently drops)."""
    auto = auto_class(config)
    return bool(getattr(auto._model_mapping[type(config)], '_supports_sdpa', False))


def load_model(name, device_map='cuda', dtype=torch.bfloat16, **kwargs):
    """The model `name` with the auto-class its config calls for, in `dtype`, asking for SDPA
    attention only from models that support it (others get their default, eager)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(name)
    if supports_sdpa(config):
        kwargs.setdefault('attn_implementation', 'sdpa')
    return auto_class(config).from_pretrained(name, dtype=dtype, device_map=device_map, **kwargs)


# -- the bridge ------------------------------------------------------------------------------

def make_bridge(name, hf_model, tokenizer, device='cuda'):
    """transformer-lens' TransformerBridge around hf_model if it supports the architecture,
    else an HFBridge: the same interface for what ChatLM uses, without hooks."""
    from transformer_lens.model_bridge import TransformerBridge
    try:
        return TransformerBridge.boot_transformers(name, hf_model=hf_model, tokenizer=tokenizer,
                                                   device=device)
    except ValueError as e:
        # transformer-lens says this one of two ways: 'Could not determine supported
        # architecture from config ...' (Inkling 2, in 4.4.0) or 'Unsupported architecture: ...'
        if 'architecture' not in str(e):
            raise
        warnings.warn(f"transformer-lens doesn't support {name} ({str(e)[:120]}...): using the "
                      "model directly (HFBridge), so no transformer-lens hooks for it")
        return HFBridge(hf_model, tokenizer)


class HFBridge:
    """
    The parts of transformer-lens' TransformerBridge that ChatLM uses, on a plain Hugging Face
    model: .tokenizer, .device, to_tokens, to_string, to_single_token, generate and a forward
    pass giving logits. For models transformer-lens doesn't support yet; no hooks.

    Generation stops at the model's own end-of-turn token(s), from its config or generation
    config: Inkling 2's tokenizer names no EOS at all, only its config does (200006,
    <|content_model_end_sampling|>).
    """

    def __init__(self, hf_model, tokenizer):
        self.hf_model, self.tokenizer = hf_model, tokenizer
        self.device = hf_model.device
        eos = None
        for src in (getattr(hf_model, 'generation_config', None), hf_model.config, tokenizer):
            eos = getattr(src, 'eos_token_id', None) if src is not None else None
            if eos is not None:
                break
        self.eos_token_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        self.eos_token_id = self.eos_token_ids[0] if self.eos_token_ids else None

    def to_tokens(self, text, prepend_bos=True):
        """(1, T) ids of `text`, on the model's device, after the BOS token if prepend_bos."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if prepend_bos:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("this tokenizer has no BOS token; use prepend_bos=False")
            ids = [bos] + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def to_string(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        elif torch.is_tensor(ids):
            ids = ids.reshape(-1).tolist()
        return self.tokenizer.decode(ids)

    def to_single_token(self, text):
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"{text!r} is {len(ids)} tokens, not one")
        return ids[0]

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=10, do_sample=False, stop_at_eos=True,
                 verbose=False, **kwargs):
        """The prompt and what follows it, (B, T + new), as TransformerBridge.generate."""
        kwargs.update(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      attention_mask=torch.ones_like(input_ids))
        if stop_at_eos and self.eos_token_ids:
            kwargs['eos_token_id'] = self.eos_token_ids
        pad = self.tokenizer.pad_token_id
        kwargs.setdefault('pad_token_id', pad if pad is not None else self.eos_token_id)
        return self.hf_model.generate(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def __call__(self, input_ids):
        """Logits (B, T, vocab), as the bridge's forward pass."""
        return self.hf_model(input_ids=input_ids).logits


# -- chat-template facts ChatLM can't read off the template ----------------------------------

# Inkling 2's template (Inkling 2 Mini and Small share it) ignores add_generation_prompt and
# enable_thinking, so ChatLM can't find either by comparing renders. A turn is written as
#   <|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>...
# and reasoning is set only by a 'Thinking effort level' system message (0.9 if not given;
# 'none' is its minimum, which per the model card doesn't switch reasoning off entirely).
_INKLING2 = dict(end_of_reasoning='<|end_message|><|message_model|><|content_text|>',
                 answer_prefix='<|message_model|><|content_text|>',
                 no_think_kwargs={'reasoning_effort': 'none'})

CHAT_SETTINGS = {'inkling2_loop': _INKLING2, 'inkling2': _INKLING2}


def chat_settings(config):
    """ChatLM keyword arguments for this model's template quirks, by model_type ({} for models
    whose template ChatLM reads on its own, like Qwen and Gemma)."""
    return dict(CHAT_SETTINGS.get(getattr(config, 'model_type', None), {}))