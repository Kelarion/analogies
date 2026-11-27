"""
The ICL experiment as a super_experiments Model, so it runs through the usual machinery:

    exp = sxp.Experiment(AutomatonTask(...), ICLModel(mode='single', think=False))
    exp.run()                       # task.sample(), then ICLModel.fit(**data)
    exp.save_experiment(SAVE_DIR)

and sweeps are the usual task_args / mod_args dicts. Everything about the run is a field
of ICLModel or of the task (so it is hashed into the save name and recorded in the arguments
sidecar). The prompt belongs to the task, which tailors it (GrammarTask.prompt, prompts.py) and
passes it on as data['prompt']; the tokens the task's symbols are written as are given by
a vocabulary.Vocabulary (also importable from icl_utils).
"""
import gc
import os
from dataclasses import dataclass, fields

import numpy as np
import torch

import util
import super_experiments as sxp
from icl_utils import ChatLM, split_block
import prompts
import vocabulary
import geometry
import categories


# -- model cache ------------------------------------------------------------------------

_BRIDGES = {}
_HF = {}

def get_lm(model_name, prepend_bos=False):
    """A ChatLM for `model_name`, loading the weights only the first time. Loading a
    different model first drops the current one, since two large models won't share a GPU
    (any `lm` you still hold elsewhere keeps the old one alive; del it first)."""
    if model_name not in _BRIDGES:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens.model_bridge import TransformerBridge

        _BRIDGES.clear(); _HF.clear()
        gc.collect(); torch.cuda.empty_cache()
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _BRIDGES[model_name] = TransformerBridge.boot_transformers(
            model_name, hf_model=hf_model, tokenizer=tokenizer, device="cuda")
        _HF[model_name] = hf_model
        # the bridge may switch this back to eager; check:
        # print(hf_model.config.get_text_config()._attn_implementation)
    lm = ChatLM(_BRIDGES[model_name], prepend_bos=prepend_bos)
    lm.hf_model = _HF[model_name]
    return lm


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
    tokens           the tokens the task symbols are written as: a vocabulary.Vocabulary,
                     a base pool and filters on frequency and lexical semantics, e.g.
                     Vocabulary('words', content=True, zipf=(3, 4)); or just a base pool's
                     name ('nouns', 'chars', 'words', 'any'). Saved as given (see vocabulary's
                     docstring for the spec strings some earlier runs used). Its base pool
                     also keys prompts.POOLS, for prompts that say what the words are.
    delim            the delimiter (task symbol 0); never in the vocabulary
    equiv            score with TokenEquivalence (e.g. ' cat' and 'cat' together)
    group            None to draw each sample's tokens uniformly from the pool; or which
                     symbols get similar tokens, as a set: 'block' (each block's own) and/or
                     kinds (each grammar token across blocks), e.g. {'block'}, {'node'}.
                     Needs the task's data['code'] and data['kinds'] (see geometry.py)
    categories       where similar tokens come from, with a group: a set of sources, e.g.
                     {'hypernyms'}, {'frames', 'hypernyms'}, {'kmeans100'}, {'semibmf300'}
                     (see geometry.py)
    scramble         a control, with a group: None, 'roles' (role groups pair mismatched
                     roles across blocks) or 'all' (tokens selected as for the group, then
                     shuffled across the shown symbols); see geometry.py
    embedding        what similarity is measured in ('input', 'canonical', or an .npz): the
                     tokens are selected by it, with a group, and every run records each
                     sample's similarities in it (metrics['similarity'], see geometry.py)
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
    tokens: object = 'nouns'
    delim: str = ','
    equiv: bool = False
    prepend_bos: bool = False
    seed: int = None
    group: object = None
    scramble: str = None
    categories: object = frozenset({'hypernyms'})
    embedding: str = 'input'

    @classmethod
    def prerequisites(cls, args):
        """Files a run with these arguments (a dict, as a sweep gives them; fields left out
        take their defaults) needs before it starts: the stored fits its geometry names, for
        its model. For server_utils.send_to_server, which makes any that are missing first."""
        default = {f.name: f.default for f in fields(cls)}
        if args.get('group', default['group']) is None:
            return []
        model = args.get('model_name', default['model_name'])
        embedding = args.get('embedding', default['embedding'])
        return [categories.prerequisite(model, c, embedding)
                for c in sorted(args.get('categories', default['categories'])) if categories.is_name(c)]

    def __post_init__(self):
        vocabulary.check_tokens(self.tokens)     # checked, never rewritten: saved as given
        geometry.check_embedding(self.embedding)
        geometry.check_categories(self.categories, self.embedding)
        geometry.check_scramble(self.scramble, self.group)
        self._geometry = (None if self.group is None else       # checks group; not an argument
                          geometry.Geometry(self.group, self.categories, self.embedding))
        super().__post_init__()

    def fit(self, blocks, X, prompt=None, **data):
        """prompt: {'spec', 'facts'} from the task (see GrammarTask); None for none."""
        if prompt is not None and self.k_show == 0 and prompts.needs_prefix(**prompt):
            raise ValueError("the hint names the start of each final trial, so it needs k_show > 0")
        lm = get_lm(self.model_name, self.prepend_bos)
        rng = np.random.default_rng(self.seed)
        pool = vocabulary.get_pool(self.tokens, lm.tokenizer, self.delim, self.equiv)
        n_tok = data['V'] if 'V' in data else len(np.unique(np.concatenate(X)))   # hidden kinds too

        template_kwargs = {} if self.effort is None else {'reasoning_effort': self.effort}
        equiv = pool.equiv

        # top_ids, top_lp: the most likely token over the whole vocabulary and over text
        # only, per row of logp_ctx (see ChatLM.run_trial); top_vocab decodes their ids.
        # logp_iso, top_ids_iso, top_lp_iso: the same, with each block scored on its own,
        # without the blocks before it or any reasoning (see ChatLM.run_isolated). kl_iso:
        # per row, KL(P ‖ Q) over the whole vocabulary from the model's prediction (P) to that
        # one (Q); the KL over the task alphabet is kl_rows(logp_ctx, logp_iso).
        # plot_icl(..., iso=...) plots either.
        # Thinking ('dialogue' and 'single'), also logp_think, top_ids_think, top_lp_think:
        # each block's assistant entry ({('assistant', b): rows}, as logp_ctx's) scored as the
        # answer to the conversation cut after that block's user turn, after reasoning of the
        # model's own (see ChatLM.answer; the final block's is logp_ctx's), and logp_nothink,
        # top_ids_nothink, top_lp_nothink: the same with the thought closed at once. No
        # reasoning trace is carried into a later block: earlier blocks are always laid out
        # as in logp_ctx, without one.
        reasons = self.think and self.mode != 'raw'
        self.metrics = {'logp_ctx': [], 'logp_iso': [], 'kl_iso': [], 'ans': [], 'task_tokens': [],
                        'prompts': [], 'top_ids': [], 'top_lp': [], 'top_ids_iso': [],
                        'top_lp_iso': [], 'top_vocab': {},
                        'data': dict(data, blocks=blocks, X=X, prompt=prompt)}
        if reasons:
            self.metrics.update({k: [] for k in ('logp_think', 'top_ids_think', 'top_lp_think',
                                                 'logp_nothink', 'top_ids_nothink', 'top_lp_nothink')})
        assigner = None
        space = geometry.get_space(self.embedding, pool, lm, self.model_name)
        self.metrics['similarity'], self.metrics['similarity_info'] = [], space.info()
        if self._geometry is not None:
            if 'code' not in data or 'kinds' not in data:
                raise ValueError("a geometry needs the task's data['code'] and data['kinds'] "
                                 "(which symbols are which grammar tokens, in which blocks)")
            assigner = geometry.get_assigner(self._geometry, pool, lm, self.model_name)
            self.metrics['geometry'] = []
            self.metrics['geometry_codes'] = assigner.cats.meta   # how any stored fits were made
        for i, blks in enumerate(blocks):
            if assigner is None:
                task_toks = pool.assign(n_tok, rng)
            else:
                code = data['code'][i]
                if self.scramble == 'roles':
                    code = geometry.scramble_roles(code, data['kinds'], rng)
                groups = geometry.symbol_groups(code, data['kinds'], self.group)
                seen = np.unique(X[i])                    # leave out symbols the trials never show
                groups = [g[np.isin(g, seen)] for g in groups]
                task_toks, info = assigner.assign(n_tok, rng, groups)
                if self.scramble == 'all':
                    task_toks, info = geometry.scramble_all(task_toks, info, seen, rng)
                self.metrics['geometry'].append(info)
            self.metrics['similarity'].append(space.cosines(task_toks).astype(np.float16))

            code = data['code'][i] if 'code' in data else None
            system_prompt, block_text, first_text = self._prompt(lm, prompt, blks, task_toks, code)
            trial_kwargs = dict(k_show=self.k_show, k_hide=self.k_hide,
                                system_prompt=system_prompt, think=self.think, equiv=equiv,
                                effort_without_thinking=self.effort_without_thinking,
                                **template_kwargs)
            lp, a, ids, top, ref = lm.run_trial(self.mode, blks, task_toks, block_text=block_text,
                                                thinking_budget=self.thinking_budget,
                                                return_hidden=True, **trial_kwargs)
            lp_iso, top_iso, kl_iso = lm.run_isolated(self.mode, blks, task_toks,
                                                      block_text=first_text, ref=ref, **trial_kwargs)
            del ref
            self.metrics['logp_iso'].append(lp_iso)
            self.metrics['top_ids_iso'].append(top_iso['ids'])
            self.metrics['top_lp_iso'].append(top_iso['lp'])
            self.metrics['kl_iso'].append({k: v.astype(np.float32) for k, v in kl_iso.items()})
            tops = [top, top_iso]
            if reasons:
                tops += self._answers(lm, blks, task_toks, block_text, lp, top, trial_kwargs)

            self.metrics['logp_ctx'].append(lp)
            self.metrics['ans'].append(a)
            self.metrics['top_ids'].append(top['ids'])
            self.metrics['top_lp'].append(top['lp'])
            seen = np.unique(np.concatenate([v.ravel() for t in tops for v in t['ids'].values()]))
            self.metrics['top_vocab'].update(
                {int(t): lm.tokenizer.decode([int(t)]) for t in seen
                 if t >= 0 and int(t) not in self.metrics['top_vocab']})
            self.metrics['task_tokens'].append(lm.tokenizer.convert_ids_to_tokens(task_toks))
            self.metrics['prompts'].append(lm.tokenizer.decode(ids))
            print(f'[{i + 1}/{len(blocks)}] done', flush=True)

    def _answers(self, lm, blks, task_toks, block_text, lp, top, trial_kwargs):
        """Each block's completion scored as the conversation's answer, with reasoning and
        without (see fit), into the metrics; returns their tops, for top_vocab."""
        last = len(blks) - 1
        out = {'think': ({}, {'ids': {}, 'lp': {}}), 'nothink': ({}, {'ids': {}, 'lp': {}})}
        for b in range(len(blks)):
            key = ('assistant', b)
            cut = dict(trial_kwargs, block_text=None if block_text is None else block_text[:b + 1])
            for name, reason in [('think', True), ('nothink', False)]:
                if reason and b == last:                  # run_trial's: no second reasoning run
                    rows, t = lp[key], {'ids': top['ids'][key], 'lp': top['lp'][key]}
                else:
                    rows, t = lm.answer(self.mode, blks[:b + 1], task_toks, reason=reason,
                                        thinking_budget=self.thinking_budget, **cut)
                out[name][0][key] = rows
                out[name][1]['ids'][key], out[name][1]['lp'][key] = t['ids'], t['lp']
        for name, (rows, t) in out.items():
            self.metrics[f'logp_{name}'].append(rows)
            self.metrics[f'top_ids_{name}'].append(t['ids'])
            self.metrics[f'top_lp_{name}'].append(t['lp'])
        return [t for _, t in out.values()]

    def _prompt(self, lm, prompt, blks, task_toks, code=None):
        """The task's prompt rendered for one sample: (system message or None, text
        before each block or None, text before each block were it the first block or None).
        The last is for scoring blocks on their own (metrics['logp_iso']): a block keeps its
        own fields but loses the transition and relabelling text, which describe a previous
        block; the system message stays the same. code: the sample's data['code'], for the
        hint's remapping."""
        if prompt is None:
            return None, None, None
        splits = [split_block(trials, self.k_show, self.k_hide) for trials in blks]
        word = lambda t: lm.tokenizer.decode([task_toks[t]]).strip()
        finals = prompts.finals_of(splits, code, word)
        render = lambda f: prompts.render(prompt['spec'], prompt['facts'], self.mode,
                                          vocabulary.base_pool(self.tokens), f)
        out = render(finals)
        system = {'role': 'system', 'content': out['system']} if out['system'] else None
        # each block in the first block's place: the system text still comes from the last
        first = [render([dict(f, remap=None)] + finals[1:])['blocks'][0] for f in finals]
        return system, out['blocks'], first