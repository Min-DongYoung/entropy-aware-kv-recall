import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AttentionSnapshot:
    alpha: torch.Tensor
    valid_positions: torch.Tensor
    attn_entropy: float
    reliability: float
    kv_len_valid: int


class KVMonitor:
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def _mean_attention(self, attentions) -> torch.Tensor:
        per_layer = []
        for layer_attn in attentions:
            if layer_attn is None:
                raise RuntimeError("Encountered None attention tensor in outputs.attentions.")
            if layer_attn.dim() != 4:
                raise RuntimeError(
                    f"Unexpected attention tensor rank={layer_attn.dim()} (expected 4)."
                )
            layer_slice = layer_attn[0, :, -1, :]
            per_layer.append(layer_slice.mean(dim=0))
        stacked = torch.stack(per_layer, dim=0)
        return stacked.mean(dim=0)

    def build_snapshot(self, attentions, attention_mask) -> AttentionSnapshot:
        if attentions is None:
            raise RuntimeError(
                "Model did not return attention weights. Ensure output_attentions=True and "
                'use an attention implementation that supports attentions (e.g., "eager").'
            )

        alpha_raw = self._mean_attention(attentions)
        kv_len = int(alpha_raw.size(0))
        valid_mask = attention_mask[0, :kv_len].bool()
        valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
        kv_len_valid = int(valid_positions.numel())

        if kv_len_valid == 0:
            return AttentionSnapshot(
                alpha=torch.empty(0, device=alpha_raw.device, dtype=alpha_raw.dtype),
                valid_positions=valid_positions,
                attn_entropy=float("nan"),
                reliability=float("nan"),
                kv_len_valid=0,
            )

        alpha = alpha_raw[valid_mask].clamp(min=0.0)
        denom = alpha.sum() + self.eps
        alpha = alpha / denom
        attn_entropy = -(alpha * torch.log(alpha + self.eps)).sum()
        if kv_len_valid <= 1:
            reliability = float("nan")
        else:
            reliability = float(1.0 - attn_entropy.item() / math.log(kv_len_valid))

        return AttentionSnapshot(
            alpha=alpha,
            valid_positions=valid_positions,
            attn_entropy=float(attn_entropy.item()),
            reliability=reliability,
            kv_len_valid=kv_len_valid,
        )


class ExperimentRunner:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda")
        self.monitor = KVMonitor(eps=args.eps)

    def _set_seed(self) -> None:
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def _load_dataset(self):
        return load_dataset(
            "AI-MO/aimo-external-prize-dataset", split="train", trust_remote_code=True
        )

    def _build_prompt(self, problem: str) -> str:
        return f"Question: {problem}\n\nSolution: <think>"

    def _load_model(self):
        torch_dtype = torch.float16 if self.args.torch_dtype == "float16" else torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch_dtype,
            device_map="cuda",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer

    def _compute_token_entropy(self, logits: torch.Tensor) -> float:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return float(entropy.item())

    def _pick_topk(self, snapshot: AttentionSnapshot) -> tuple[set, int]:
        if snapshot.kv_len_valid <= 0:
            return set(), 0

        if self.args.topk_mode == "ratio":
            k = max(1, int(self.args.topk_ratio * snapshot.kv_len_valid))
        else:
            k = min(self.args.topk_k, snapshot.kv_len_valid)

        topk_rel = torch.topk(snapshot.alpha, k, dim=-1).indices
        topk_abs = snapshot.valid_positions.index_select(0, topk_rel)
        return set(topk_abs.tolist()), k

    def _jaccard_drift(self, prev_set: set, cur_set: set) -> float:
        if prev_set is None:
            return float("nan")
        union = prev_set | cur_set
        if not union:
            return float("nan")
        inter = prev_set & cur_set
        return float(1.0 - len(inter) / len(union))

    def _kl_drift(self, prev_alpha: torch.Tensor, cur_alpha: torch.Tensor) -> float:
        if prev_alpha is None:
            return float("nan")
        prefix_len = min(int(prev_alpha.numel()), int(cur_alpha.numel()))
        if prefix_len <= 0:
            return float("nan")
        p = prev_alpha[:prefix_len]
        q = cur_alpha[:prefix_len]
        p = p / (p.sum() + self.args.eps)
        q = q / (q.sum() + self.args.eps)
        kl = (p * torch.log((p + self.args.eps) / (q + self.args.eps))).sum()
        return float(kl.item())

    def _forward_step(self, model, input_ids, attention_mask, past_key_values):
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
        if outputs.attentions is None:
            raise RuntimeError(
                "Attentions are None. Ensure attn_implementation='eager' and the model "
                "supports output_attentions."
            )
        return outputs

    def _summarize_phase1(self, records: list, output_dir: str) -> dict:
        df = pd.DataFrame(records)
        if df.empty:
            summary = {"skipped": True, "reason": "no_step_records"}
        else:
            df["next_drift_set"] = df.groupby("sample_id")["drift_set"].shift(-1)
            df["next_drift_kl"] = df.groupby("sample_id")["drift_kl"].shift(-1)

            def safe_corr(series_a, series_b, method: str) -> float:
                pair = pd.concat([series_a, series_b], axis=1).dropna()
                if pair.shape[0] < 2:
                    return float("nan")
                return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))

            summary = {
                "skipped": False,
                "counts": {"rows": int(df.shape[0])},
                "correlations": {
                    "token_entropy_vs_next_drift_set": {
                        "pearson": safe_corr(
                            df["token_entropy"], df["next_drift_set"], "pearson"
                        ),
                        "spearman": safe_corr(
                            df["token_entropy"], df["next_drift_set"], "spearman"
                        ),
                    },
                    "token_entropy_vs_next_drift_kl": {
                        "pearson": safe_corr(
                            df["token_entropy"], df["next_drift_kl"], "pearson"
                        ),
                        "spearman": safe_corr(
                            df["token_entropy"], df["next_drift_kl"], "spearman"
                        ),
                    },
                },
            }

        summary_path = os.path.join(output_dir, "phase1_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print("Phase 1 summary:")
        print(json.dumps(summary, indent=2))
        print(f"Saved phase1_summary.json to {summary_path}")
        return summary

    def run(self) -> None:
        self._set_seed()
        os.makedirs(self.args.output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.args.output_dir, f"phase1_{timestamp}.jsonl")

        model, tokenizer = self._load_model()
        dataset = self._load_dataset()
        total_samples = min(len(dataset), self.args.max_samples)

        meta = {
            "type": "meta",
            "timestamp": timestamp,
            "model_path": self.args.model_path,
            "dataset": "AI-MO/aimo-external-prize-dataset",
            "split": "train",
            "max_seq_len": self.args.max_seq_len,
            "max_new_tokens": self.args.max_new_tokens,
            "attn_implementation": "eager",
            "topk_mode": self.args.topk_mode,
            "topk_ratio": self.args.topk_ratio,
            "topk_k": self.args.topk_k,
            "eps": self.args.eps,
            "seed": self.args.seed,
            "torch_dtype": self.args.torch_dtype,
        }

        step_records = []
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(json.dumps(meta, ensure_ascii=True) + "\n")

            for sample_idx, sample in enumerate(
                tqdm(dataset, total=total_samples, desc="Samples", unit="sample")
            ):
                if sample_idx >= self.args.max_samples:
                    break

                problem = sample.get("problem", "")
                prompt = self._build_prompt(problem)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.args.max_seq_len,
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                seq_len = int(input_ids.size(1))
                if seq_len == 0:
                    continue

                max_new_tokens = min(
                    self.args.max_new_tokens, self.args.max_seq_len - seq_len
                )
                if max_new_tokens <= 0:
                    continue

                past_key_values = None
                prev_alpha = None
                prev_set = None

                with torch.no_grad():
                    for idx in range(seq_len - 1):
                        token = input_ids[:, idx : idx + 1]
                        current_mask = attention_mask[:, : idx + 1]
                        outputs = self._forward_step(model, token, current_mask, past_key_values)
                        past_key_values = outputs.past_key_values

                    current_token = input_ids[:, seq_len - 1 : seq_len]
                    current_mask = attention_mask[:, :seq_len]

                    for step in range(max_new_tokens):
                        outputs = self._forward_step(
                            model, current_token, current_mask, past_key_values
                        )
                        logits = outputs.logits[:, -1, :]
                        token_entropy = self._compute_token_entropy(logits)

                        snapshot = self.monitor.build_snapshot(
                            outputs.attentions, current_mask
                        )
                        cur_set, k_used = self._pick_topk(snapshot)
                        drift_set = self._jaccard_drift(prev_set, cur_set)
                        drift_kl = self._kl_drift(prev_alpha, snapshot.alpha)

                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        token_id = int(next_token.item())
                        token_str = tokenizer.decode([token_id])

                        record = {
                            "type": "step",
                            "sample_id": sample_idx,
                            "step": step,
                            "token_id": token_id,
                            "token_str": token_str,
                            "token_entropy": token_entropy,
                            "attn_entropy": snapshot.attn_entropy,
                            "reliability": snapshot.reliability,
                            "drift_set": drift_set,
                            "drift_kl": drift_kl,
                            "kv_len_valid": snapshot.kv_len_valid,
                            "K_used": k_used,
                        }
                        log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                        step_records.append(record)

                        past_key_values = outputs.past_key_values
                        prev_alpha = snapshot.alpha.detach()
                        prev_set = cur_set

                        if (
                            self.args.stop_on_eos
                            and tokenizer.eos_token_id is not None
                            and token_id == int(tokenizer.eos_token_id)
                        ):
                            break

                        if current_mask.size(1) + 1 > self.args.max_seq_len:
                            break

                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    (1, 1), device=self.device, dtype=attention_mask.dtype
                                ),
                            ],
                            dim=1,
                        )
                        current_mask = attention_mask
                        current_token = next_token

                log_file.flush()

        print(f"Saved logs to {log_path}")
        self._summarize_phase1(step_records, self.args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 drift experiment.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-8B-Thinking",
        choices=[
            "Qwen/Qwen3-8B-Thinking",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ],
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch_dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--topk_mode", type=str, choices=["ratio", "fixed"], default="ratio")
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument("--topk_k", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--stop_on_eos", action="store_true")
    args = parser.parse_args()

    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
