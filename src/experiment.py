import argparse
import json
import os
import random
import inspect
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


BUDGET_RATIOS = (0.1, 0.2, 0.5)
RECALL_KEYS = {0.1: "recall_10", 0.2: "recall_20", 0.5: "recall_50"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_entropy(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def build_prompt(example: dict) -> str:
    if "input" in example and isinstance(example["input"], str):
        return example["input"]
    context = example.get("context", "")
    question = example.get("question", "")
    if isinstance(context, list):
        context = "\n".join(context)
    if isinstance(question, list):
        question = " ".join(question)
    if context and question:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    if question:
        return f"Question: {question}\nAnswer:"
    if "prompt" in example and isinstance(example["prompt"], str):
        return example["prompt"]
    return str(example)


def extract_layerwise_attention(outputs) -> torch.Tensor:
    if outputs.attentions is None:
        # Flash Attention 2 등 일부 구현체는 output_attentions=True를 지원하지 않음
        raise RuntimeError(
            "Model did not return attention weights. "
            "Please ensure `attn_implementation='eager'` is used, "
            "as Flash Attention does not support returning attention weights."
        )
    per_layer = []
    for layer_attn in outputs.attentions:
        if layer_attn is None:
            raise RuntimeError("Encountered None attention tensor in outputs.attentions.")
        if layer_attn.dim() != 4:
            raise RuntimeError(
                f"Unexpected attention tensor rank={layer_attn.dim()} (expected 4)."
            )
        # [Batch, Heads, Q_len, KV_len] -> [Batch, KV_len] (Sum over heads)
        # Q_len is assumed to be 1 (incremental decoding)
        collapsed = layer_attn[0].sum(dim=0)
        if collapsed.size(0) != 1:
            raise RuntimeError(
                f"Expected q_len=1 for incremental decoding, got q_len={collapsed.size(0)}."
            )
        per_layer.append(collapsed.squeeze(0).to(dtype=torch.float32))
    return torch.stack(per_layer, dim=0)


def compute_recalls(
    attn_per_layer: torch.Tensor,
    accum_scores_per_layer: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple,
    window_size: int,
) -> dict:
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_count = int(valid_positions.numel())
    recalls = {}
    if valid_count == 0:
        for ratio in budget_ratios:
            recalls[ratio] = 0.0
        return recalls

    num_layers = int(attn_per_layer.size(0))
    for ratio in budget_ratios:
        k = max(1, int(ratio * valid_count))
        k = min(k, valid_count)
        layer_recalls = []
        cutoff_idx = max(0, valid_count - window_size)
        window_positions = valid_positions[cutoff_idx:]
        history_positions = valid_positions[:cutoff_idx]

        for layer_idx in range(num_layers):
            scores_gold_valid = attn_per_layer[layer_idx].index_select(0, valid_positions)
            gold_rel = torch.topk(scores_gold_valid, k, dim=-1).indices
            gold_pos = valid_positions.index_select(0, gold_rel)

            remaining_k = k - int(window_positions.numel())
            if remaining_k > 0:
                hist_scores = accum_scores_per_layer[layer_idx].index_select(
                    0, history_positions
                )
                hh_rel = torch.topk(hist_scores, remaining_k, dim=-1).indices
                hh_pos = history_positions.index_select(0, hh_rel)
                pred_pos = torch.cat([hh_pos, window_positions], dim=0)
            else:
                pred_pos = window_positions[-k:]

            hits = torch.isin(pred_pos, gold_pos).sum().to(dtype=torch.float32)
            layer_recalls.append(hits / float(k))

        recalls[ratio] = float(torch.stack(layer_recalls).mean().item())
    return recalls


def forward_step(model, input_ids, attention_mask, past_key_values):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.clamp_min(0)[:, -1:]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    attn_per_layer = extract_layerwise_attention(outputs)
    return outputs, attn_per_layer


def get_flash_attn_version() -> str:
    try:
        import flash_attn
        return flash_attn.__version__
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Entropy-aware KV cache recall experiment.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--no_stop_on_eos", action="store_true")
    parser.add_argument("--window_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"{args.dataset_name}_{timestamp}.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # [중요] Flash Attention 2 -> Eager 모드로 변경
        trust_remote_code=True,
    )
    model.eval()

    # 데이터셋 로딩 (견고성 강화)
    load_dataset_kwargs = {}
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        load_dataset_kwargs["trust_remote_code"] = True

    try:
        dataset = load_dataset(
            "THUDM/LongBench",
            args.dataset_name,
            split=args.dataset_split,
            **load_dataset_kwargs,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "Failed to load `THUDM/LongBench`. Please verify `datasets` version (e.g. 2.19.0)."
            ) from exc
        raise

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        try:
            num_layers = int(len(model.model.layers))
        except Exception as exc:
            raise RuntimeError("Unable to infer the number of transformer layers.") from exc

    meta = {
        "type": "meta",
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "max_seq_len": args.max_seq_len,
        "attn_implementation": "eager",  # 메타데이터 업데이트
        "flash_attn_version": get_flash_attn_version(),
        "window_size": args.window_size,
        "num_layers": num_layers,
    }

    device = torch.device("cuda")
    eos_token_id = tokenizer.eos_token_id
    stop_on_eos = not args.no_stop_on_eos

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(json.dumps(meta, ensure_ascii=True) + "\n")

        for sample_idx, example in enumerate(dataset):
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break

            prompt = build_prompt(example)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = int(input_ids.size(1))
            if seq_len == 0:
                continue

            max_new_tokens = min(args.max_new_tokens, args.max_seq_len - seq_len)
            if max_new_tokens <= 0:
                continue

            accum_scores = torch.zeros(
                (num_layers, args.max_seq_len), device=device, dtype=torch.float32
            )
            past_key_values = None

            with torch.no_grad():
                # [Prefill 단계]
                for idx in range(seq_len - 1):
                    token = input_ids[:, idx : idx + 1]
                    current_mask = attention_mask[:, : idx + 1]
                    outputs, attn_per_layer = forward_step(
                        model, token, current_mask, past_key_values
                    )
                    kv_len = int(attn_per_layer.size(1))
                    accum_scores[:, :kv_len] += attn_per_layer
                    past_key_values = outputs.past_key_values

                current_token = input_ids[:, seq_len - 1 : seq_len]
                current_mask = attention_mask[:, :seq_len]

                # [Generation 단계]
                for step in range(max_new_tokens):
                    outputs, attn_per_layer = forward_step(
                        model, current_token, current_mask, past_key_values
                    )
                    logits = outputs.logits[:, -1, :]
                    entropy = compute_entropy(logits)
                    kv_len = int(attn_per_layer.size(1))
                    valid_mask = current_mask[0, :kv_len].bool()
                    recalls = compute_recalls(
                        attn_per_layer,
                        accum_scores,
                        valid_mask,
                        BUDGET_RATIOS,
                        args.window_size,
                    )

                    record = {
                        "type": "step",
                        "sample_id": sample_idx,
                        "step": step,
                        "entropy": entropy,
                        RECALL_KEYS[0.1]: recalls[0.1],
                        RECALL_KEYS[0.2]: recalls[0.2],
                        RECALL_KEYS[0.5]: recalls[0.5],
                    }
                    log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    if step % args.log_every == 0:
                        log_file.flush()

                    accum_scores[:, :kv_len] += attn_per_layer
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    if stop_on_eos and eos_token_id is not None:
                        if int(next_token.item()) == int(eos_token_id):
                            break

                    if current_mask.size(1) + 1 > args.max_seq_len:
                        break
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
                        ],
                        dim=1,
                    )
                    current_mask = attention_mask
                    current_token = next_token

            log_file.flush()

    print(f"Saved logs to {log_path}")


if __name__ == "__main__":
    main()