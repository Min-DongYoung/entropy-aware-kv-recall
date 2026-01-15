import argparse
import json
import math
import os
import random
import time
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


def extract_last_layer_attention(outputs) -> torch.Tensor:
    if outputs.attentions is None:
        raise RuntimeError(
            "Model did not return attention weights. Ensure output_attentions=True "
            "and that the model supports attention outputs with flash_attention_2."
        )
    last_layer = outputs.attentions[-1]
    attn = last_layer[0].mean(dim=0).squeeze(0)
    return attn.float()


def compute_recalls(
    attn_vec: torch.Tensor,
    accum_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple,
    window_size: int,
) -> dict:
    valid_count = int(valid_mask.sum().item())
    recalls = {}
    if valid_count == 0:
        for ratio in budget_ratios:
            recalls[ratio] = 0.0
        return recalls

    scores_gold = attn_vec.clone()
    scores_pred = accum_scores[: attn_vec.size(0)].clone()
    scores_gold[~valid_mask] = -float("inf")
    scores_pred[~valid_mask] = -float("inf")

    for ratio in budget_ratios:
        k = max(1, int(ratio * valid_count))
        k = min(k, valid_count)
        gold_idx = torch.topk(scores_gold, k, dim=-1).indices
        cutoff_idx = max(0, valid_count - window_size)
        window_indices = torch.arange(cutoff_idx, valid_count, device=attn_vec.device)
        remaining_k = k - int(window_indices.numel())

        if remaining_k > 0:
            hist_scores = scores_pred[:cutoff_idx]
            if hist_scores.numel() > 0:
                hh_indices = torch.topk(hist_scores, remaining_k, dim=-1).indices
                pred_idx = torch.cat([hh_indices, window_indices], dim=0)
            else:
                pred_idx = window_indices
        else:
            pred_idx = window_indices[-k:]
        hits = torch.isin(pred_idx, gold_idx).sum().item()
        recalls[ratio] = hits / k
    return recalls


def forward_step(model, input_ids, attention_mask, past_key_values):
    position_ids = attention_mask.cumsum(-1)[:, -1:]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    attn_vec = extract_last_layer_attention(outputs)
    return outputs, attn_vec


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
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()

    dataset = load_dataset("THUDM/LongBench", args.dataset_name, split=args.dataset_split)

    meta = {
        "type": "meta",
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "max_seq_len": args.max_seq_len,
        "attn_implementation": "flash_attention_2",
        "flash_attn_version": get_flash_attn_version(),
        "window_size": args.window_size,
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

            accum_scores = torch.zeros(args.max_seq_len, device=device, dtype=torch.float32)
            past_key_values = None

            with torch.no_grad():
                for idx in range(seq_len - 1):
                    token = input_ids[:, idx : idx + 1]
                    current_mask = attention_mask[:, : idx + 1]
                    outputs, attn_vec = forward_step(
                        model, token, current_mask, past_key_values
                    )
                    kv_len = attn_vec.size(0)
                    accum_scores[:kv_len] += attn_vec
                    past_key_values = outputs.past_key_values

                current_token = input_ids[:, seq_len - 1 : seq_len]
                current_mask = attention_mask[:, :seq_len]

                for step in range(max_new_tokens):
                    outputs, attn_vec = forward_step(
                        model, current_token, current_mask, past_key_values
                    )
                    logits = outputs.logits[:, -1, :]
                    entropy = compute_entropy(logits)
                    kv_len = attn_vec.size(0)
                    valid_mask = current_mask[0, :kv_len].bool()
                    recalls = compute_recalls(
                        attn_vec,
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

                    accum_scores[:kv_len] += attn_vec
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
