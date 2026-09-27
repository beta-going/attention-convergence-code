"""
Re-run Protocol I probe with LONG, punctuation-dense input text.
This should reveal punctuation-segmentation attention patterns that are
invisible in short sequences.

IMPORTANT: This is a TEMPLATE. You need to adapt the model loading and
attention extraction to match your existing probe pipeline.

Usage:
    python rerun_probe_long_input.py --output-dir ./probe_v3_long
"""

import argparse
import json
import os

import numpy as np
import os, yaml
_CFG = yaml.safe_load(open(
    os.environ.get("AE_MODELS_CONFIG", "configs/models.yaml")))
def _mp(k):
    m = _CFG["models"][k]
    return m.get("local_path") or m["hf_id"]


PROBE_TEXT_ZH = (
    "人工智能的发展经历了多个阶段。首先，符号主义学派认为智能可以通过逻辑推理实现；"
    "然而，这种方法在处理不确定性时遇到了困难。接着，连接主义学派提出了神经网络模型，"
    "它通过大量数据的训练来学习模式。但是，深度学习需要巨大的计算资源！此外，"
    "强化学习为智能体提供了与环境交互的能力。总之，每种方法都有其优缺点。"
    "近年来，大语言模型的出现改变了整个领域的格局。例如，GPT系列模型展示了惊人的能力；"
    "不过，它们也带来了新的挑战：幻觉问题、对齐问题、以及安全性问题。"
    "未来，我们需要在效率、可靠性和可控性之间找到平衡。"
    "从技术角度来看，Transformer架构是当前最成功的方案。它的核心机制是自注意力，"
    "即每个token都可以关注序列中的所有其他位置。这种设计使得模型能够捕获长距离依赖关系。"
    "然而，标准自注意力的计算复杂度是序列长度的二次方！这意味着处理长文本时，"
    "内存和计算成本会急剧增加。为了解决这个问题，研究者提出了多种优化方案："
    "稀疏注意力、线性注意力、以及滑动窗口注意力等。每种方案都在效率和表达能力之间做出了不同的权衡。"
)

PROBE_TEXT_EN = (
    "The development of artificial intelligence has gone through several stages. "
    "First, the symbolic approach assumed that intelligence could be achieved through logical reasoning; "
    "however, this method encountered difficulties when dealing with uncertainty. "
    "Next, the connectionist school proposed neural network models, which learn patterns through training on large datasets. "
    "But deep learning requires enormous computational resources! Moreover, reinforcement learning provides agents with the ability to interact with their environment. "
    "In summary, each approach has its own strengths and weaknesses. "
    "In recent years, the emergence of large language models has transformed the entire field. "
    "For example, the GPT series demonstrated remarkable capabilities; however, they also introduced new challenges: hallucination, alignment, and safety concerns. "
    "Going forward, we need to find a balance between efficiency, reliability, and controllability. "
    "From a technical perspective, the Transformer architecture is currently the most successful approach. "
    "Its core mechanism is self-attention, where each token can attend to all other positions in the sequence. "
    "This design enables the model to capture long-range dependencies. "
    "However, the computational complexity of standard self-attention is quadratic in sequence length! "
    "This means that when processing long texts, memory and computation costs increase dramatically. "
    "To address this issue, researchers have proposed various optimization strategies: sparse attention, linear attention, and sliding window attention. "
    "Each strategy makes different trade-offs between efficiency and expressiveness."
)

MODEL_CONFIGS = [
    {
        "name": "Qwen3-4B",
        "path": _mp("qwen3-4b"),  # ← CHANGE THIS
        "texts": {"zh": PROBE_TEXT_ZH, "en": PROBE_TEXT_EN},
    },
    {
        "name": "Qwen3-8B",
        "path": _mp("qwen3-8b"),  # ← CHANGE THIS
        "texts": {"zh": PROBE_TEXT_ZH, "en": PROBE_TEXT_EN},
    },
    {
        "name": "Llama-3.1-8B",
        "path": _mp("llama-3.1-8b"),  # ← CHANGE THIS
        "texts": {"zh": PROBE_TEXT_ZH, "en": PROBE_TEXT_EN},
    },
    {
        "name": "Mistral-7B",
        "path": _mp("mistral-7b"),  # ← CHANGE THIS
        "texts": {"zh": PROBE_TEXT_ZH, "en": PROBE_TEXT_EN},
    },
]

def extract_attention_vllm(model_path, text, n_layers=None):
    """
    Extract per-head attention using vLLM or transformers.
    
    Returns:
        dict with keys 'layer_0', 'layer_1', ... each containing
        numpy array of shape [n_heads, seq_len, seq_len]
        Also includes 'tokens' list for punctuation verification.
    
    NOTE: Adapt this function to your existing probe pipeline!
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"  Tokenized length: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (n_layers,) tensors
    # each tensor: [batch=1, n_heads, seq_len, seq_len]
    attentions = outputs.attentions
    n_layers_actual = len(attentions)

    result = {
        "n_layers": n_layers_actual,
        "n_heads": attentions[0].shape[1],
        "seq_len": seq_len,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist()),
    }

    for L in range(n_layers_actual):
        attn = attentions[L][0].cpu().float().numpy()  # [n_heads, seq_len, seq_len]
        result[f"layer_{L}"] = attn.tolist()

    # Cleanup
    del model, outputs, attentions
    torch.cuda.empty_cache()

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../data/attention")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        path = cfg["path"]

        if not os.path.isdir(path):
            print(f"⚠️  Skipping {name}: path '{path}' not found. Update MODEL_CONFIGS.")
            continue

        for lang, text in cfg["texts"].items():
            out_file = os.path.join(args.output_dir, f"consistency_{name}_{lang}.json")
            print(f"\n{'='*60}")
            print(f"Processing: {name} | lang={lang}")
            print(f"  Text preview: {text[:80]}...")

            result = extract_attention_vllm(path, text)

            # Add metadata
            result["model"] = name
            result["lang"] = lang
            result["input_text"] = text

            with open(out_file, "w") as f:
                json.dump(result, f)

            print(f"  ✅ Saved: {out_file}")
            print(f"     Layers: {result['n_layers']}, Heads: {result['n_heads']}, "
                  f"SeqLen: {result['seq_len']}")

            # Show punctuation positions
            tokens = result["tokens"]
            punct_chars = set("，。！？、；：,.!?;:")
            punct_positions = [i for i, t in enumerate(tokens) if any(c in t for c in punct_chars)]
            print(f"     Punctuation positions: {punct_positions}")

    print(f"\n🎉 Done! All files saved to {args.output_dir}/")
    print("Next step: run scan_all_layers.py on the new directory to find segmentation heads.")
    
if __name__ == "__main__":
    main()