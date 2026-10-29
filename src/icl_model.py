"""
The ICL experiment as a super_experiments Model, so it runs through the usual machinery:

    exp = sxp.Experiment(AutomatonTask(...), ICLModel(mode='single', think=False))
    exp.run()                       # task.sample(), then ICLModel.fit(**data)
    exp.save_experiment(SAVE_DIR)

and sweeps are the usual task_args / mod_args dicts. Everything about the run is a field
of ICLModel or of the task (so it is hashed into the save name and recorded in the arguments
sidecar). The prompt belongs to the task, which tailors it (GrammarTask.prompt, prompts.py) and
passes it on as data['prompt']; token pools are referred to by name, from the registry below.
"""
import gc
import os
from dataclasses import dataclass

import numpy as np
import torch

import util
import super_experiments as sxp
from icl_utils import ChatLM, TokenEquivalence, split_block
import prompts

# -- registries ---------------------------------------------------------------------

def _nouns(lm):
    with open(os.path.join(os.path.dirname(util.__file__), 'top-1000-nouns.txt')) as f:
        return lm.single_tokens([w.strip() for w in f], leading_space=True)


def _chars(lm):
    import string
    return lm.single_tokens(string.ascii_letters + string.digits, leading_space=False)


TOKEN_POOLS = {'nouns': _nouns, 'chars': _chars}


# -- model cache ------------------------------------------------------------------------

_BRIDGES = {}


def get_lm(model_name, prepend_bos=False):
    """A ChatLM for `model_name`, loading the weights only the first time. Loading a
    different model first drops the current one, since two large models won't share a GPU
    (any `lm` you still hold elsewhere keeps the old one alive; del it first)."""
    if model_name not in _BRIDGES:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens.model_bridge import TransformerBridge

        _BRIDGES.clear()
        gc.collect()
        torch.cuda.empty_cache()

        hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                        device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _BRIDGES[model_name] = TransformerBridge.boot_transformers(
            model_name, hf_model=hf_model, tokenizer=tokenizer, device="cuda")
    return ChatLM(_BRIDGES[model_name], prepend_bos=prepend_bos)


# -- the model ----------------------------------------------------------------------------

@dataclass
class ICLModel(sxp.Model):
    """
    model_name       Hugging Face id of the weights
    mode             'dialogue' | 'single' | 'raw' (see ChatLM.build_prompt)
    think            let the model reason before answering (ignored in 'raw' mode)
    k_show, k_hide   how much of each completion is shown / held out (see split_block)
    effort           reasoning_effort passed to the chat template, None to leave it out
    effort_without_thinking  add the template's effort instruction even when think=False
    thinking_budget  max reasoning tokens
    tokens           key into TOKEN_POOLS: the words the task symbols are mapped onto
                     (and into prompts.POOLS, for prompts that say what the words are)
    equiv            score with TokenEquivalence (e.g. ' cat' and 'cat' together)
    prepend_bos      start 'raw' prompts with the BOS token
    seed             seed for the symbol-to-word assignment of each sample
    """
    model_name: str = "Qwen/Qwen3.8-27B"
    mode: str = 'dialogue'
    think: bool = True
    k_show: int = None
    k_hide: int = 1
    effort: str = None
    effort_without_thinking: bool = False
    thinking_budget: int = 4000
    tokens: str = 'nouns'
    delim: str = ','
    equiv: bool = False
    prepend_bos: bool = False
    seed: int = None

    def fit(self, blocks, X, prompt=None, **data):
        """prompt: {'spec', 'facts'} from the task (see GrammarTask); None for none."""
        if prompt is not None and self.k_show == 0 and prompts.needs_prefix(**prompt):
            raise ValueError("the hint names the start of each final trial, so it needs k_show > 0")
        lm = get_lm(self.model_name, self.prepend_bos)
        rng = np.random.default_rng(self.seed)
        pool = TOKEN_POOLS[self.tokens](lm)
        n_tok = len(np.unique(np.concatenate(X)))

        template_kwargs = {} if self.effort is None else {'reasoning_effort': self.effort}
        equiv = TokenEquivalence(lm.tokenizer) if self.equiv else None

        self.metrics = {'logp_ctx': [], 'ans': [], 'task_tokens': [], 'prompts': [],
                        'data': dict(data, blocks=blocks, X=X, prompt=prompt)}
        for i, blks in enumerate(blocks):
            task_toks = rng.permutation(pool)[:n_tok]
            task_toks[0] = lm.bridge.to_single_token(self.delim)

            system_prompt, block_text = self._prompt(lm, prompt, blks, task_toks)
            lp, a, ids = lm.run_trial(
                self.mode, blks, task_toks, k_show=self.k_show, k_hide=self.k_hide,
                system_prompt=system_prompt, think=self.think,
                thinking_budget=self.thinking_budget, equiv=equiv, block_text=block_text,
                effort_without_thinking=self.effort_without_thinking, **template_kwargs)

            self.metrics['logp_ctx'].append(lp)
            self.metrics['ans'].append(a)
            self.metrics['task_tokens'].append(lm.tokenizer.convert_ids_to_tokens(task_toks))
            self.metrics['prompts'].append(lm.tokenizer.decode(ids))
            print(f'[{i + 1}/{len(blocks)}] done', flush=True)

    def _prompt(self, lm, prompt, blks, task_toks):
        """The task's prompt rendered for one sample: (system message or None, text
        before each block or None)."""
        if prompt is None:
            return None, None
        finals = []                                   # each block's final trial
        for trials in blks:
            ctx, comp = split_block(trials, self.k_show, self.k_hide)
            cut = np.where(ctx == 0)[0]
            shown = ctx[cut[-1] + 1:] if len(cut) else ctx
            finals.append(([lm.tokenizer.decode([t]).strip() for t in task_toks[shown]],
                           len(comp)))
        out = prompts.render(prompt['spec'], prompt['facts'], self.mode, self.tokens, finals)
        system = {'role': 'system', 'content': out['system']} if out['system'] else None
        return system, out['blocks']