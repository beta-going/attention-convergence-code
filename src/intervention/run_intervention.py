#!/usr/bin/env python3
"""
Attention-intervention syntactic probes.

CODE VERSION: v8.0 (2026-09)
Engine: FORKED FROM v6.0 (2026-08-31) — the code that produced all published
numbers (paper v25/v27). v6 stays frozen and untouched; this file only adds an
engineering shell around the identical engine.

HISTORY NOTE: an earlier, unrelated "v8" (output-space surrogate engine) was
aborted before producing any data and has been fully deleted. The v8 version
number is reused here for the v6-fork.

Changes vs v6.0 (engineering shell only — engine numerics are bit-identical):
  [1] Crash-safe incremental logging: stream.jsonl records one line per probe
      and per generation sample as it is produced (plus a final summary line).
  [2] ALL generation samples are saved (v6 truncated to first 3).
  [3] Optional automatic ZH probe filtering (--zh_auto_filter): keeps probes
      with baseline log-prob gap > 0.5 (same criterion as paper Appendix N).
      Manual --zh_probe_filter still works and takes priority.
  [4] Baseline gate: reports baseline cloze accuracy before the mode loop;
      aborts only with --strict_baseline_gate (EN is not expected to be 100%
      per the paper protocol; auto-filtered ZH is 100% by construction).
  [5] --self_test: standalone engine validation — hooks fire, logits change
      under intervention, bit-exact restore after unpatch, and
      uniform_causal output == weight-space replacement reference.
  [6] Loader flags: --trust_remote_code, --torch_dtype, --bnb_compute_dtype
      (Mixtral wants bfloat16), plus sliding-window sanity check.
  [7] Rich metadata: env fingerprint (transformers/torch versions), operator
      identity ("value_space_exact_v6"), shuffle permutation semantics,
      generation semantics, special-token policy.
  [8] --unify_special_tokens: sensitivity-analysis only. Default OFF preserves
      v6 comparability (v6: cloze without BOS, free-form/generation with BOS
      for Llama/Mistral).

NOT changed (byte-identical to v6.0): all probe datasets, InterventionManager
surrogates (including shuffle sharing ONE permutation across all intervened
layers for a given seed), compute_layer_indices, evaluation math, evaluation
default special-token policy, generation semantics (cached greedy generation
+ full-length teacher-forced PPL forward, intervention active in both).

Usage:
  # Engine self-test (run once after writing this file)
  python src/intervention_experiment_v8.py --model $QWEN_06B \
      --layer_split early --self_test

  # Anchor reproduction (compare against paper Table 4 / Table 6:
  # U-Causal 140/99.8, U-Global 108/110, Shuffle 261/299 EN/ZH)
  python src/intervention_experiment_v8.py --model $QWEN_06B \
      --layer_split early \
      --modes baseline,uniform_causal,uniform_global,shuffle_rows \
      --output_dir results_v8/anchor_qwen06b_early

  # New model: InternLM3-8B-Instruct (check exact checkpoint name; there is
  # no upstream id "internlm-8b")
  python src/intervention_experiment_v8.py --model $INTERNLM_8B \
      --layer_split early \
      --modes baseline,uniform_causal,uniform_global,shuffle_rows \
      --trust_remote_code --zh_auto_filter \
      --output_dir results_v8/internlm3-8b_early

  # New model: Qwen3-8B-Base (instruct/base controlled pair with Qwen3-8B)
  python src/intervention_experiment_v8.py --model $QWEN_8B_BASE \
      --layer_split early \
      --modes baseline,uniform_causal,uniform_global,shuffle_rows \
      --zh_auto_filter --output_dir results_v8/qwen3-8b-base_early

  # New model: Llama-3.1-8B-Base (second instruct/base pair)
  python src/intervention_experiment_v8.py --model $LLAMA_8B_BASE \
      --layer_split early \
      --modes baseline,uniform_causal,uniform_global,shuffle_rows \
      --zh_auto_filter --output_dir results_v8/llama-3.1-8b-base_early

  # Mixtral-8x7B-Instruct on server (INT4, bf16 compute)
  python src/intervention_experiment_v8.py \
      --model /root/data/models/Mixtral-8x7B-Instruct-v0.1 \
      --layer_split early \
      --modes baseline,uniform_causal,uniform_global,shuffle_rows \
      --load_in_4bit --bnb_compute_dtype bfloat16 --trust_remote_code \
      --zh_auto_filter --output_dir results_v8/mixtral_early
"""

import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_VERSION = "v8.0"
ENGINE_VERSION = "v6.0-fork (numerics identical)"

# Global evaluation-policy switches (default = v6 behavior).
SETTINGS = {
    "cloze_special_tokens": False,   # v6: cloze tokenization without BOS
}

MASK = "{BLANK}"

# ======================================================================
# Probes — IDENTICAL to v3.0/v6.0 (not a single character changed)
# ======================================================================
EN_CLOZE = [
    # --- Subject-Verb Agreement (4) ---
    {"prompt": f"The cat {MASK} on the mat.", "options": ["sits", "sit"], "correct": 0},
    {"prompt": f"The dogs {MASK} in the yard.", "options": ["runs", "run"], "correct": 1},
    {"prompt": f"She {MASK} to school every day.", "options": ["goes", "go"], "correct": 0},
    {"prompt": f"They {MASK} happy today.", "options": ["was", "were"], "correct": 1},
    # --- Tense (3) ---
    {"prompt": f"Yesterday, he {MASK} to the store.", "options": ["walked", "walks"], "correct": 0},
    {"prompt": f"Tomorrow, she will {MASK} home.", "options": ["go", "goes"], "correct": 0},
    {"prompt": f"Right now, they are {MASK} lunch.", "options": ["eating", "ate"], "correct": 0},
    # --- Articles (3) ---
    {"prompt": f"I saw {MASK} elephant at the zoo.", "options": ["an", "a"], "correct": 0},
    {"prompt": f"She is {MASK} teacher at our school.", "options": ["a", "an"], "correct": 0},
    {"prompt": f"He ate {MASK} apple that was on the table.", "options": ["the", "a"], "correct": 0},
    # --- Prepositions (3) ---
    {"prompt": f"The book is {MASK} the table.", "options": ["on", "in"], "correct": 0},
    {"prompt": f"She arrived {MASK} the airport at noon.", "options": ["at", "on"], "correct": 0},
    {"prompt": f"He is interested {MASK} music.", "options": ["in", "on"], "correct": 0},
    # --- Pronoun Case (2) ---
    {"prompt": f"I gave the book to {MASK}.", "options": ["him", "he"], "correct": 0},
    {"prompt": f"She and {MASK} went to the store.", "options": ["I", "me"], "correct": 0},
    # --- Pluralization (2) ---
    {"prompt": f"There are many {MASK} in the classroom.", "options": ["children", "childs"], "correct": 0},
    {"prompt": f"The {MASK} ran away from the cat.", "options": ["mice", "mouses"], "correct": 0},
    # --- Comparative (1) ---
    {"prompt": f"This box is {MASK} than that one.", "options": ["heavier", "more heavy"], "correct": 0},
    # --- Auxiliary / Modal (2) ---
    {"prompt": f"He {MASK} swim very well.", "options": ["can", "cans"], "correct": 0},
    {"prompt": f"She {MASK} like coffee.", "options": ["doesn't", "don't"], "correct": 0},
]

EN_FREE = [
    "The quick brown fox jumps over the lazy dog.",
    "She has been studying English for three years.",
    "If I had known earlier, I would have told you.",
    "The children are playing in the park.",
    "He doesn't know where she went.",
    "There are many reasons why this happened.",
    "The book that I read yesterday was fascinating.",
    "They have already finished their homework.",
    "She is taller than her brother.",
    "We should have left earlier to avoid the traffic.",
]

ZH_CLOZE = [
    # --- 主谓一致 (3) ---
    {"prompt": f"学生们都{MASK}了教室。", "options": ["离开", "离开了"], "correct": 0},
    {"prompt": f"他每天{MASK}去上班。", "options": ["骑车", "骑车子"], "correct": 0},
    {"prompt": f"我{MASK}喜欢这本书。", "options": ["非常", "很非常"], "correct": 0},
    # --- 的/地/得 (3) ---
    {"prompt": f"美丽{MASK}花朵开了。", "options": ["的", "地"], "correct": 0},
    {"prompt": f"他高兴{MASK}跳了起来。", "options": ["地", "的"], "correct": 0},
    {"prompt": f"跑{MASK}很快。", "options": ["得", "的"], "correct": 0},
    # --- 了/着/过 (2) ---
    {"prompt": f"我{MASK}经吃过饭了。", "options": ["已", "以"], "correct": 0},
    {"prompt": f"他正{MASK}看书呢。", "options": ["在", "再"], "correct": 0},
    # --- 把/被/让 (3) ---
    {"prompt": f"他{MASK}书放在桌子上。", "options": ["把", "被"], "correct": 0},
    {"prompt": f"杯子{MASK}他打碎了。", "options": ["被", "把"], "correct": 0},
    {"prompt": f"老师{MASK}我们做作业。", "options": ["让", "使"], "correct": 0},
    # --- 量词 (2) ---
    {"prompt": f"一{MASK}书放在桌上。", "options": ["本", "个"], "correct": 0},
    {"prompt": f"三{MASK}猫在睡觉。", "options": ["只", "个"], "correct": 0},
    # --- 连词 (2) ---
    {"prompt": f"{MASK}然下雨了，我们还是去了。", "options": ["虽", "随"], "correct": 0},
    {"prompt": f"{MASK}为下雨，所以取消了。", "options": ["因", "应"], "correct": 0},
    # --- 否定与双重否定 (2) ---
    {"prompt": f"他不可能不{MASK}道。", "options": ["知", "之"], "correct": 0},
    {"prompt": f"没有人不{MASK}欢这首歌。", "options": ["喜", "洗"], "correct": 0},
    # --- 比较结构 (1) ---
    {"prompt": f"他比我{MASK}。", "options": ["高", "更加高"], "correct": 0},
    # --- 是/在 (2) ---
    {"prompt": f"这{MASK}我的书。", "options": ["是", "在"], "correct": 0},
    {"prompt": f"他{MASK}家里等你。", "options": ["在", "是"], "correct": 0},
]

ZH_FREE = [
    "虽然今天下着大雨，但是他还是按时到达了公司。",
    "这本书的内容非常丰富，值得反复阅读。",
    "如果明天天气好的话，我们就去公园散步吧。",
    "他不仅学习好，而且体育也很棒。",
    "这个问题比较复杂，需要仔细分析才能解决。",
    "她说的话让我感到非常惊讶和意外。",
    "尽管遇到了很多困难，他们依然坚持完成了任务。",
    "这个地方的风景很美，吸引了很多游客。",
    "他的态度让人觉得很不舒服。",
    "我们需要更多的时间来准备这次重要的会议。",
]

# Generation prompts
EN_GEN_PROMPTS = [
    "The student who studied hard",
    "If she had arrived earlier,",
    "The books on the shelf",
    "Neither the teacher nor the students",
    "Having finished his homework,",
]

ZH_GEN_PROMPTS = [
    "虽然今天天气不好，但是",
    "如果明天不下雨的话，",
    "这本书的内容非常",
    "他不仅会说英语，而且",
    "尽管遇到了很多困难，",
]

# ======================================================================
# Proportional layer computation — IDENTICAL to v6.0
# ======================================================================
def compute_layer_indices(num_layers: int, split: str) -> list:
    """
    early = first 25%, mid = middle 50%, late = last 25%, all = everything.
    N=28: early=[0..6], mid=[7..20], late=[21..27]
    N=32: early=[0..7], mid=[8..23], late=[24..31]
    N=36: early=[0..8], mid=[9..26], late=[27..35]
    N=40: early=[0..9], mid=[10..29], late=[30..39]
    """
    if split == "all":
        return list(range(num_layers))
    q1 = int(num_layers * 0.25)
    q3 = int(num_layers * 0.75)
    if split == "early":
        return list(range(0, q1))
    elif split == "mid":
        return list(range(q1, q3))
    elif split == "late":
        return list(range(q3, num_layers))
    else:
        raise ValueError(f"Unknown layer_split: {split}. Use early/mid/late/all.")

# ======================================================================
# Intervention Manager — IDENTICAL to v6.0 (engine core, do not modify)
# ======================================================================
class InterventionManager:
    """
    Modes:
      baseline        -> no intervention
      uniform_causal  -> position i attends equally to positions 0..i
      uniform_global  -> every position attends equally to all positions
      shuffle_rows    -> random permutation of the output sequence dimension
                         (ONE permutation shared across all intervened layers
                          for a given seed — v6 semantics, preserved)
    """

    def __init__(self, model, layers: list, mode: str, seed: int = 42):
        self.model = model
        self.layers = layers
        self.mode = mode
        self.seed = seed
        self.call_counts = {i: 0 for i in layers}
        self.handles = []
        
    def _matched_global(self, module, args, kwargs, attn_output):
        from matched_global import matched_global_mean
        V = self._get_value_states(module, args, kwargs)
        Vmg = matched_global_mean(V, dim=-2)
        n_rep = module.o_proj.in_features // Vmg.shape[-1]
        return module.o_proj(Vmg.repeat_interleave(n_rep, dim=-1))

    # ---- lifecycle ----
    def activate(self):
        for layer_idx in self.layers:
            attn_module = self.model.model.layers[layer_idx].self_attn
            h = attn_module.register_forward_hook(
                self._make_hook(layer_idx), with_kwargs=True
            )
            self.handles.append(h)
        print(f"[Intervention] Registered {len(self.handles)} hooks on "
              f"layers {self.layers}, mode={self.mode}")

    def deactivate(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.call_counts = {i: 0 for i in self.layers}

    def verify(self):
        bad = [i for i, c in self.call_counts.items() if c == 0]
        if bad:
            raise RuntimeError(
                f"[{CODE_VERSION}] Hooks on layers {bad} NEVER fired. "
                "Check attn_implementation='eager' and module path. ABORTING."
            )
        total = sum(self.call_counts.values())
        print(f"[Verify] All hooks fired. Total calls: {total}, "
              f"per-layer: {self.call_counts}")

    # ---- hook factory ----
    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            self.call_counts[layer_idx] += 1

            if self.mode == 'baseline':
                return output

            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            if self.mode == 'uniform_causal':
                new_out = self._uniform_causal(module, args, kwargs, attn_output)
            elif self.mode == 'uniform_global':
                new_out = self._uniform_global(module, args, kwargs, attn_output)
            elif self.mode == 'shuffle_rows':
                new_out = self._shuffle_rows(attn_output)
            elif self.mode == 'matched_global':
                new_out = self._matched_global(module, args, kwargs, attn_output)
            else:
                return output

            if isinstance(output, tuple):
                return (new_out,) + output[1:]
            return new_out
        return hook

    def _get_value_states(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        V = module.v_proj(hidden_states)
        return V

    @staticmethod
    def _get_attn_config(module):
        # --- num_heads ---
        num_heads = getattr(module, 'num_heads', None)
        if num_heads is None:
            cfg = getattr(module, 'config', None)
            if cfg is not None:
                num_heads = getattr(cfg, 'num_attention_heads', None)
            if num_heads is None:
                raise AttributeError(
                    f"Cannot determine num_heads on {type(module).__name__}. "
                    f"Module attrs: {[a for a in dir(module) if not a.startswith('_')]}"
                )
        # --- head_dim ---
        head_dim = getattr(module, 'head_dim', None)
        if head_dim is None:
            cfg = getattr(module, 'config', None)
            if cfg is not None:
                head_dim = getattr(cfg, 'head_dim', None)
            if head_dim is None:
                hidden_size = getattr(module, 'hidden_size', None)
                if hidden_size is None and cfg is not None:
                    hidden_size = getattr(cfg, 'hidden_size', None)
                if hidden_size is not None:
                    head_dim = hidden_size // num_heads
            if head_dim is None:
                raise AttributeError(
                    f"Cannot determine head_dim on {type(module).__name__}"
                )
        # --- num_kv_heads ---
        num_kv_heads = getattr(module, 'num_key_value_heads', None)
        if num_kv_heads is None:
            num_kv_groups = getattr(module, 'num_key_value_groups', None)
            if num_kv_groups is not None:
                num_kv_heads = num_heads // num_kv_groups
            else:
                cfg = getattr(module, 'config', None)
                if cfg is not None:
                    num_kv_heads = getattr(cfg, 'num_key_value_heads', None)
                if num_kv_heads is None:
                    num_kv_heads = num_heads
        return num_heads, head_dim, num_kv_heads

    def _uniform_causal(self, module, args, kwargs, attn_output):
        """
        output[i] = mean(V[0..i])  ==  o_proj( W_uniform_causal @ V )
        with W[t,j] = 1/(t+1) for j<=t  (exact weight-space replacement).
        """
        V = self._get_value_states(module, args, kwargs)
        B, q_len, kv_dim = V.shape
        num_heads, head_dim, num_kv_heads = self._get_attn_config(module)
        num_kv_groups = num_heads // num_kv_heads

        V = V.view(B, q_len, num_kv_heads, head_dim)
        V_cumsum = V.cumsum(dim=1)
        counts = torch.arange(1, q_len + 1, device=V.device, dtype=V.dtype)
        counts = counts.view(1, q_len, 1, 1)
        V_uniform = V_cumsum / counts

        if num_kv_groups > 1:
            V_uniform = V_uniform.repeat_interleave(num_kv_groups, dim=2)
        V_uniform = V_uniform.reshape(B, q_len, num_heads * head_dim)
        new_out = module.o_proj(V_uniform)
        return new_out

    def _uniform_global(self, module, args, kwargs, attn_output):
        """
        output[i] = mean(V[0..T-1])  ==  o_proj( W_uniform_global @ V )
        with W[t,j] = 1/T  (exact weight-space replacement).
        """
        V = self._get_value_states(module, args, kwargs)
        B, q_len, kv_dim = V.shape
        num_heads, head_dim, num_kv_heads = self._get_attn_config(module)
        num_kv_groups = num_heads // num_kv_heads

        V = V.view(B, q_len, num_kv_heads, head_dim)
        V_mean = V.mean(dim=1, keepdim=True)

        if num_kv_groups > 1:
            V_mean = V_mean.repeat_interleave(num_kv_groups, dim=2)
        V_mean = V_mean.expand(B, q_len, num_heads, head_dim)
        V_mean = V_mean.reshape(B, q_len, num_heads * head_dim)
        new_out = module.o_proj(V_mean)
        return new_out

    def _shuffle_rows(self, attn_output):
        """Seeded permutation of the sequence dimension (v6 semantics)."""
        q_len = attn_output.shape[1]
        g = torch.Generator(device=attn_output.device)
        g.manual_seed(self.seed)
        perm = torch.randperm(q_len, device=attn_output.device, generator=g)
        return attn_output[:, perm, :].clone()

# ======================================================================
# Input preparation — IDENTICAL to v6.0 (v6.2 fix retained)
# ======================================================================
def _prepare_inputs(inputs, device):
    LONG_KEYS = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device).clone()
            if k in LONG_KEYS:
                v = v.long()
        result[k] = v
    return result

# ======================================================================
# [NEW v8.0] Stream logger — crash-safe incremental JSONL
# ======================================================================
class StreamLogger:
    """One JSON line per record; everything before a crash is preserved."""

    def __init__(self, output_dir: str, meta: dict):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "stream.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")
        self.write({"type": "meta", **meta})
        print(f"[Stream] Incremental log -> {self.path}")

    def write(self, rec: dict):
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

# ======================================================================
# Model loading — v6.0 + [NEW v8.0] dtype/trust_remote_code/SWA check
# ======================================================================
def load_model(path, load_in_4bit=False, torch_dtype_str="float16",
               bnb_compute_dtype_str="float16", trust_remote_code=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tdtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16
    bdtype = torch.bfloat16 if bnb_compute_dtype_str == "bfloat16" else torch.float16

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bdtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=trust_remote_code,
        )
        print(f"[Load] INT4 quantization (NF4, double quant, compute={bnb_compute_dtype_str})")
    else:
        try:
            # for big models,on server, because there is RTX4090 24G x2,120G Memory,device_map=device_map="auto",and offload to cpu
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=tdtype,
                device_map={"": 0},
                attn_implementation="eager",
                trust_remote_code=trust_remote_code,
            )
            print(f"[Load] {torch_dtype_str} on single GPU (cuda:0)")
        except torch.cuda.OutOfMemoryError:
            print(f"[Load] Single GPU OOM, falling back to device_map='auto'")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=tdtype,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=trust_remote_code,
            )
    model.eval()

    # Sliding-window sanity check (Mistral/Mixtral: window=4096, probes<=512 -> OK)
    sw = getattr(model.config, "sliding_window", None)
    if sw:
        if sw < 512:
            print(f"[WARN] sliding_window={sw} < 512: uniform/global surrogates are "
                  f"defined over the full sequence but attention beyond the window "
                  f"is truncated by the architecture. Sequences kept <= {sw} tokens "
                  f"by truncation; results under the window are unaffected.")
        else:
            print(f"[Load] sliding_window={sw} (no conflict with probe lengths <=512)")

    return model, tok

# ======================================================================
# Evaluation functions — numerics IDENTICAL to v6.0
# ([NEW v8.0] _option_logprob reads SETTINGS["cloze_special_tokens"])
# ======================================================================
@torch.no_grad()
def _option_logprob(model, tok, prefix: str, option: str) -> float:
    add_special = SETTINGS["cloze_special_tokens"]  # default False = v6 behavior
    prefix_ids = (tok(prefix, add_special_tokens=add_special)["input_ids"]
                  if prefix.strip() else [])
    full_text = prefix + option
    full_ids = tok(full_text, add_special_tokens=add_special)["input_ids"]
    option_ids = full_ids[len(prefix_ids):]
    if len(option_ids) == 0:
        return 0.0

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    log_probs = F.log_softmax(logits, dim=-1)
    total_lp = 0.0
    for k, tid in enumerate(option_ids):
        pos = len(prefix_ids) + k - 1
        if pos < 0:
            vocab_size = model.config.vocab_size
            total_lp += -math.log(vocab_size)
        else:
            total_lp += log_probs[pos, tid].item()
    return total_lp

@torch.no_grad()
def evaluate_cloze(model, tok, probe):
    prefix = probe["prompt"].split(MASK, 1)[0]
    scores = [_option_logprob(model, tok, prefix, o) for o in probe["options"]]
    pred = int(torch.argmax(torch.tensor(scores)).item())
    return {
        "prompt": probe["prompt"],
        "options": probe["options"],
        "logprobs": scores,
        "correct": probe["correct"] == pred,
        "predicted": probe["options"][pred],
    }

@torch.no_grad()
def compute_perplexity(model, tok, text, max_length=512):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    if inputs.input_ids.size(1) < 2:
        return float("nan")
    inputs = _prepare_inputs(inputs, model.device)
    out = model(**inputs, labels=inputs["input_ids"])
    return float(math.exp(out.loss.item()))

@torch.no_grad()
def evaluate_generation(model, tok, prompts: list, max_new_tokens=60):
    results = []
    for prompt in prompts:
        inputs = _prepare_inputs(tok(prompt, return_tensors="pt"), model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        generated_ids = outputs[0][input_len:]
        generated_text = tok.decode(generated_ids, skip_special_tokens=True)

        full_ids = outputs[0].unsqueeze(0)
        full_logits = model(full_ids).logits
        shift_logits = full_logits[0, input_len - 1:-1, :]
        shift_labels = full_ids[0, input_len:]
        loss_fn = nn.CrossEntropyLoss(ignore_index=tok.eos_token_id)
        loss = loss_fn(shift_logits, shift_labels)
        ppl = float(torch.exp(loss).item()) if loss < 20 else float('inf')

        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'gen_ppl': ppl,
            'gen_length': len(generated_ids),
        })
    return results

@torch.no_grad()
def diagnose_zh_probes(model, tok, cloze_probes):
    print("\n" + "=" * 60)
    print("ZH PROBE DIAGNOSIS")
    print("=" * 60)
    ambiguous = []
    for i, p in enumerate(cloze_probes):
        prefix = p["prompt"].split(MASK, 1)[0]
        print(f"  [DBG] Probe {i:2d}: prefix={repr(prefix)}", flush=True)
        inputs = _prepare_inputs(tok(prefix, return_tensors="pt"), model.device)
        if inputs["input_ids"].numel() == 0:
            print(f"  [!] Probe {i:2d}: SKIPPED (empty token sequence)")
            continue
        logits = model(**inputs).logits[0, -1]
        a_id = tok.encode(" " + p["options"][0], add_special_tokens=False)[-1]
        b_id = tok.encode(" " + p["options"][1], add_special_tokens=False)[-1]
        lp_a = logits[a_id].item()
        lp_b = logits[b_id].item()
        diff = lp_a - lp_b
        predicted = 0 if lp_a > lp_b else 1
        is_correct = (predicted == p["correct"])
        status = "✓" if is_correct else "✗"
        amb_flag = "  ⚠️ AMBIGUOUS" if abs(diff) < 0.5 else ""
        print(f"  [{status}] Probe {i:2d}: lp(A)={lp_a:+.3f}, lp(B)={lp_b:+.3f}, "
              f"Δ={diff:+.3f}, pred={'A' if predicted == 0 else 'B'}, "
              f"gold={'A' if p['correct'] == 0 else 'B'}{amb_flag}")
        if not is_correct or abs(diff) < 0.5:
            ambiguous.append({
                'idx': i, 'probe': p, 'lp_a': lp_a, 'lp_b': lp_b,
                'diff': diff, 'predicted': predicted
            })
    if ambiguous:
        print(f"\n  ⚠️ {len(ambiguous)} probes are ambiguous or incorrect at baseline.")
    else:
        print("\n  ✓ All probes are clear and correct at baseline.")
    return ambiguous

# ======================================================================
# [NEW v8.0] Automatic ZH probe filtering (paper Appendix N criterion)
# ======================================================================
@torch.no_grad()
def auto_filter_zh(model, tok, zh_cloze, threshold=0.5):
    kept, meta = [], []
    for i, p in enumerate(zh_cloze):
        prefix = p["prompt"].split(MASK, 1)[0]
        scores = [_option_logprob(model, tok, prefix, o) for o in p["options"]]
        others = [s for j, s in enumerate(scores) if j != p["correct"]]
        gap = scores[p["correct"]] - (max(others) if others else float("-inf"))
        ok = gap > threshold
        meta.append({"idx": i, "gap": float(gap), "kept": ok})
        if ok:
            kept.append(p)
    print(f"[ZH-Filter] kept {len(kept)}/{len(zh_cloze)} probes (Δlogp > {threshold}). "
          f"Kept indices: {[m['idx'] for m in meta if m['kept']]}")
    return kept, meta

# ======================================================================
# [NEW v8.0] Baseline gate (warn by default; abort only with --strict_baseline_gate)
# ======================================================================
@torch.no_grad()
def baseline_gate(model, tok, probe_sets: dict, strict=False):
    report = {}
    for name, probes_ in probe_sets.items():
        failed = [i for i, p in enumerate(probes_)
                  if not evaluate_cloze(model, tok, p)["correct"]]
        acc = 100.0 * (len(probes_) - len(failed)) / max(1, len(probes_))
        report[name] = {"n": len(probes_), "failed": failed, "acc": acc}
        status = "PASS" if not failed else ("ABORT" if strict else "WARN (continuing)")
        print(f"[Gate] {name.upper()} baseline acc={acc:.1f}% "
              f"({len(failed)} failed: {failed}) → {status}")
    if strict and any(r["failed"] for r in report.values()):
        raise RuntimeError("[Gate] --strict_baseline_gate: baseline < 100%. ABORTING.")
    return report

# ======================================================================
# Probe evaluation bundle — IDENTICAL to v6.0
# ======================================================================
def evaluate_probes(model, tok, cloze, free):
    choice = [evaluate_cloze(model, tok, p) for p in cloze]
    acc = 100.0 * sum(c["correct"] for c in choice) / max(1, len(choice))
    free_records = [{"text": t, "ppl": compute_perplexity(model, tok, t)} for t in free]
    ppls = [r["ppl"] for r in free_records if r["ppl"] == r["ppl"]]
    return {
        "choice_probes": choice,
        "accuracy": acc,
        "n_choice": len(choice),
        "free_form_ppls": ppls,
        "free_form_records": free_records,
        "mean_ppl": (sum(ppls) / len(ppls)) if ppls else float("nan"),
    }

# ======================================================================
# [NEW v8.0] Engine self-test
# ======================================================================
@torch.no_grad()
def self_test(model, tok, target_layers, seed=42):
    if not target_layers:
        raise RuntimeError("[SelfTest] need at least one target layer")
    L = target_layers[0]
    layer = model.model.layers[L].self_attn
    H, Dh, KVH = InterventionManager._get_attn_config(layer)
    G = H // KVH

    ids = tok("The quick brown fox jumps over the lazy dog and keeps running.",
              return_tensors="pt").input_ids[:, :12].to(model.device)
    base = model(ids).logits.clone()

    # (1) hooks fire / logits change / bit-exact restore, for each surrogate
    for mode in ["uniform_causal", "uniform_global", "shuffle_rows"]:
        mgr = InterventionManager(model, [L], mode, seed=seed)
        mgr.activate()
        out = model(ids).logits
        assert all(c > 0 for c in mgr.call_counts.values()), f"{mode}: hook not fired"
        assert (out - base).abs().max().item() > 1e-3, f"{mode}: logits unchanged?!"
        mgr.deactivate()
        restored = model(ids).logits
        assert torch.equal(restored, base), f"{mode}: restore not bit-exact"
        print(f"  [SelfTest] {mode:15s} hooks fire ✓  logits change ✓  restore bit-exact ✓")

    # (2) uniform_causal == weight-space replacement reference (independent impl)
    mgr = InterventionManager(model, [L], "uniform_causal", seed=seed)
    mgr.activate()
    captured = {}
    def grab_out(module, args, kwargs, output):
        captured["attn"] = (output[0] if isinstance(output, tuple) else output).detach()
        return output
    h = layer.register_forward_hook(grab_out, with_kwargs=True)  # AFTER mgr -> patched output
    model(ids)
    h.remove()
    mgr.deactivate()

    hidden = {}
    def grab_in(module, args, kwargs):
        hidden["h"] = args[0] if args else kwargs.get("hidden_states")
    hp = layer.register_forward_pre_hook(grab_in, with_kwargs=True)
    model(ids)
    hp.remove()

    V = layer.v_proj(hidden["h"]).view(1, -1, KVH, Dh).float()
    T = V.shape[1]
    W = torch.tril(torch.ones(T, T, device=V.device)) / \
        torch.arange(1, T + 1, device=V.device).float().view(-1, 1)
    Vr = V.repeat_interleave(G, dim=2)
    x = torch.einsum("tj,bjhd->bthd", W, Vr).reshape(1, T, H * Dh)
    ref = layer.o_proj(x.to(layer.o_proj.weight.dtype))


    diff = (captured["attn"].float() - ref.float()).abs().max().item()
    assert torch.allclose(captured["attn"].float(), ref.float(), atol=1e-2, rtol=1e-2), \
        f"uniform_causal deviates from weight-space reference (max|Δ|={diff})"
    print(f"  [SelfTest] uniform_causal ≡ weight-space replacement ✓ (max|Δ|={diff:.2e})")
    print(f"[SelfTest] PASS — engine is the v6 value-space surrogate (exact).")

# ======================================================================
# Experiment driver — v6.0 + [NEW v8.0] logging / filtering / gate / meta
# ======================================================================
def run_single(model, tok, target_layers, modes, zh_cloze,
               run_diagnosis=False, run_generation=True, seed=42,
               logger=None, run_idx=0):
    if run_diagnosis:
        diagnose_zh_probes(model, tok, zh_cloze)

    results = {}
    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"  CONDITION: {mode} (seed={seed})")
        print(f"{'=' * 50}")

        mgr = None
        if mode != 'baseline':
            mgr = InterventionManager(model, target_layers, mode, seed=seed)
            mgr.activate()

        en = evaluate_probes(model, tok, EN_CLOZE, EN_FREE)
        zh = evaluate_probes(model, tok, zh_cloze, ZH_FREE)

        gen_en, gen_zh = None, None
        if run_generation:
            gen_en = evaluate_generation(model, tok, EN_GEN_PROMPTS)
            gen_zh = evaluate_generation(model, tok, ZH_GEN_PROMPTS)

        if mgr is not None:
            mgr.verify()
            mgr.deactivate()

        # [NEW v8.0] per-item incremental records (ALL generation samples)
        if logger is not None:
            for i, c in enumerate(en["choice_probes"]):
                logger.write({"type": "cloze", "mode": mode, "run": run_idx,
                              "lang": "en", "probe_idx": i, **c})
            for i, c in enumerate(zh["choice_probes"]):
                logger.write({"type": "cloze", "mode": mode, "run": run_idx,
                              "lang": "zh", "probe_idx": i, **c})
            for i, r in enumerate(en["free_form_records"]):
                logger.write({"type": "free_ppl", "mode": mode, "run": run_idx,
                              "lang": "en", "sample_idx": i, **r})
            for i, r in enumerate(zh["free_form_records"]):
                logger.write({"type": "free_ppl", "mode": mode, "run": run_idx,
                              "lang": "zh", "sample_idx": i, **r})

            if run_generation:
                for i, g in enumerate(gen_en):
                    logger.write({"type": "gen", "mode": mode, "run": run_idx,
                                  "lang": "en", "sample_idx": i, **g})
                for i, g in enumerate(gen_zh):
                    logger.write({"type": "gen", "mode": mode, "run": run_idx,
                                  "lang": "zh", "sample_idx": i, **g})

        entry = {"syntactic_en": en, "syntactic_zh": zh}
        if run_generation:
            entry["generation_en"] = {
                "avg_ppl": sum(r['gen_ppl'] for r in gen_en) / len(gen_en),
                "samples": gen_en,          # [NEW v8.0] full samples (v6 kept only 3)
            }
            entry["generation_zh"] = {
                "avg_ppl": sum(r['gen_ppl'] for r in gen_zh) / len(gen_zh),
                "samples": gen_zh,
            }
        results[mode] = entry

        print(f"  EN acc={en['accuracy']:.1f}% ppl={en['mean_ppl']:.2f}")
        print(f"  ZH acc={zh['accuracy']:.1f}% ppl={zh['mean_ppl']:.2f}")
        if run_generation:
            print(f"  EN gen_ppl={entry['generation_en']['avg_ppl']:.2f}")
            print(f"  ZH gen_ppl={entry['generation_zh']['avg_ppl']:.2f}")
    return results

def run_experiment(model_path, target_layers, modes, output_dir,
                   run_diagnosis=False, run_generation=True,
                   zh_probe_filter=None, zh_auto_filter=False,
                   load_in_4bit=False, num_repeats=1,
                   torch_dtype_str="float16", bnb_compute_dtype_str="float16",
                   trust_remote_code=False, strict_baseline_gate=False,
                   unify_special_tokens=False):
    import transformers
    SETTINGS["cloze_special_tokens"] = unify_special_tokens

    model, tok = load_model(model_path, load_in_4bit=load_in_4bit,
                            torch_dtype_str=torch_dtype_str,
                            bnb_compute_dtype_str=bnb_compute_dtype_str,
                            trust_remote_code=trust_remote_code)

    # ---- ZH probe selection: manual override > auto filter > all (v6 default) ----
    zh_filter_meta = None
    zh_cloze = ZH_CLOZE
    if zh_probe_filter is not None:
        indices = [int(x) for x in zh_probe_filter.split(",")]
        zh_cloze = [ZH_CLOZE[i] for i in indices]
        print(f"[ProbeFilter] manual: using {len(zh_cloze)}/{len(ZH_CLOZE)} ZH probes: {indices}")
    elif zh_auto_filter:
        zh_cloze, zh_filter_meta = auto_filter_zh(model, tok, ZH_CLOZE, threshold=0.5)

    # ---- baseline gate (report + meta; aborts only in strict mode) ----
    gate_report = baseline_gate(
        model, tok, {"en": EN_CLOZE, "zh": zh_cloze},
        strict=strict_baseline_gate)

    num_total_layers = len(model.model.layers)
    meta = {
        "code_version": CODE_VERSION,
        "engine": ENGINE_VERSION,
        "model": model_path,
        "num_total_layers": num_total_layers,
        "intervention_layers": list(target_layers),
        "layer_split_info": f"{len(target_layers)} layers out of {num_total_layers} "
                            f"({100*len(target_layers)/num_total_layers:.1f}%)",
        "modes": modes,
        "timestamp": datetime.now().isoformat(),
        "zh_probe_filter": zh_probe_filter,
        "zh_auto_filter": zh_auto_filter,
        "zh_filter_meta": zh_filter_meta,
        "baseline_gate": gate_report,
        "load_in_4bit": load_in_4bit,
        "num_repeats": num_repeats,
        "n_en_cloze": len(EN_CLOZE),
        "n_zh_cloze": len(zh_cloze),
        "n_en_free": len(EN_FREE),
        "n_zh_free": len(ZH_FREE),
        # [NEW v8.0] environment / semantics fingerprint
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "operator": "value_space_exact_v6",
        "shuffle_perm_semantics": "one permutation shared across all intervened layers per seed",
        "gen_semantics": "cached greedy generation (intervention at prefill) + "
                         "full-length teacher-forced PPL forward (intervention at every position)",
        "special_tokens_policy": ("unified (BOS everywhere)" if unify_special_tokens
                                  else "v6: cloze w/o BOS; free-form & generation with BOS"),
    }

    os.makedirs(output_dir, exist_ok=True)
    logger = StreamLogger(output_dir, meta)

    all_results = {"meta": meta}
    for rep in range(num_repeats):
        seed = 42 + rep
        print(f"\n{'#' * 60}")
        print(f"  REPEAT {rep+1}/{num_repeats} (seed={seed})")
        print(f"{'#' * 60}")
        diag = run_diagnosis if rep == 0 else False
        results = run_single(
            model, tok, target_layers, modes, zh_cloze,
            run_diagnosis=diag, run_generation=run_generation,
            seed=seed, logger=logger, run_idx=rep,
        )
        if num_repeats == 1:
            all_results.update(results)
        else:
            all_results[f"repeat_{rep}"] = results

    # ---- summary (mean ± std) ----
    if num_repeats > 1:
        summary = {}
        for mode in modes:
            mode_data = {}
            for metric_lang in ["syntactic_en", "syntactic_zh"]:
                accs = [all_results[f"repeat_{r}"][mode][metric_lang]["accuracy"]
                        for r in range(num_repeats)]
                ppls = [all_results[f"repeat_{r}"][mode][metric_lang]["mean_ppl"]
                        for r in range(num_repeats)]
                m = sum(accs) / len(accs)
                mode_data[f"{metric_lang}_acc_mean"] = m
                mode_data[f"{metric_lang}_acc_std"] = (
                    sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5
                mode_data[f"{metric_lang}_ppl_mean"] = sum(ppls) / len(ppls)
            if run_generation:
                for gen_lang in ["generation_en", "generation_zh"]:
                    gppls = [all_results[f"repeat_{r}"][mode][gen_lang]["avg_ppl"]
                             for r in range(num_repeats)]
                    m = sum(gppls) / len(gppls)
                    mode_data[f"{gen_lang}_ppl_mean"] = m
                    mode_data[f"{gen_lang}_ppl_std"] = (
                        sum((p - m) ** 2 for p in gppls) / len(gppls)) ** 0.5
            summary[mode] = mode_data
        all_results["summary"] = summary
        logger.write({"type": "summary", **summary})
        print(f"\n{'=' * 60}\nSUMMARY (mean ± std across repeats)\n{'=' * 60}")
        for mode in modes:
            s = summary[mode]
            print(f"  {mode:20s} | EN acc={s['syntactic_en_acc_mean']:.1f}"
                  f"±{s['syntactic_en_acc_std']:.1f}%"
                  f" ZH acc={s['syntactic_zh_acc_mean']:.1f}"
                  f"±{s['syntactic_zh_acc_std']:.1f}%")
    else:
        logger.write({"type": "summary",
                      **{m: {"en_acc": all_results[m]["syntactic_en"]["accuracy"],
                             "zh_acc": all_results[m]["syntactic_zh"]["accuracy"],
                             **({"en_gen_ppl": all_results[m]["generation_en"]["avg_ppl"],
                                 "zh_gen_ppl": all_results[m]["generation_zh"]["avg_ppl"]}
                                if run_generation else {})}
                         for m in modes}})
    logger.close()

    out_path = os.path.join(output_dir, "intervention_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")

# ======================================================================
# CLI — v6.0 + [NEW v8.0] flags
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description=f"Attention intervention experiment ({CODE_VERSION}, engine v6-fork)"
    )
    ap.add_argument("--model", required=True, help="Path to model directory")

    layer_group = ap.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--intervention_layers",
                             help="Comma-separated layer indices")
    layer_group.add_argument("--layer_split", choices=["early", "mid", "late", "all"],
                             help="Proportional split: early(25%%)/mid(50%%)/late(25%%)/all")

    ap.add_argument("--modes", default="baseline,uniform_causal,shuffle_rows",
                    help="Comma-separated: baseline, uniform_causal, "
                         "uniform_global, shuffle_rows")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--diagnose_zh", action="store_true")
    ap.add_argument("--no_generation", action="store_true")
    ap.add_argument("--zh_probe_filter", default=None,
                    help="Manual comma-separated indices (takes priority over auto filter)")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--num_repeats", type=int, default=1)

    # [NEW v8.0]
    ap.add_argument("--self_test", action="store_true",
                    help="Run engine self-test on the first target layer, then exit")
    ap.add_argument("--zh_auto_filter", action="store_true",
                    help="Auto-filter ZH probes with baseline Δlogp > 0.5 "
                         "(paper Appendix N criterion)")
    ap.add_argument("--strict_baseline_gate", action="store_true",
                    help="Abort if any probe fails at baseline (default: warn only)")
    ap.add_argument("--unify_special_tokens", action="store_true",
                    help="Sensitivity analysis: add BOS to cloze tokenization too. "
                         "Default OFF preserves v6 comparability.")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--bnb_compute_dtype", default="float16",
                    choices=["float16", "bfloat16"])

    args = ap.parse_args()

    if args.intervention_layers:
        layers = [int(x) for x in args.intervention_layers.split(",")]
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model,
                                            trust_remote_code=args.trust_remote_code)
        num_layers = config.num_hidden_layers
        layers = compute_layer_indices(num_layers, args.layer_split)
        print(f"[LayerSplit] Model has {num_layers} layers, split='{args.layer_split}' "
              f"→ layers {layers[0]}..{layers[-1]} ({len(layers)} layers)")

    modes = [m.strip() for m in args.modes.split(",")]

    if args.self_test:
        model, tok = load_model(args.model,
                                load_in_4bit=args.load_in_4bit,
                                torch_dtype_str=args.torch_dtype,
                                bnb_compute_dtype_str=args.bnb_compute_dtype,
                                trust_remote_code=args.trust_remote_code)
        self_test(model, tok, layers, seed=42)
        print("[SelfTest] done — exiting without running the experiment.")
        return

    run_experiment(
        model_path=args.model,
        target_layers=layers,
        modes=modes,
        output_dir=args.output_dir,
        run_diagnosis=args.diagnose_zh,
        run_generation=not args.no_generation,
        zh_probe_filter=args.zh_probe_filter,
        zh_auto_filter=args.zh_auto_filter,
        load_in_4bit=args.load_in_4bit,
        num_repeats=args.num_repeats,
        torch_dtype_str=args.torch_dtype,
        bnb_compute_dtype_str=args.bnb_compute_dtype,
        trust_remote_code=args.trust_remote_code,
        strict_baseline_gate=args.strict_baseline_gate,
        unify_special_tokens=args.unify_special_tokens,
    )

if __name__ == "__main__":
    main()
