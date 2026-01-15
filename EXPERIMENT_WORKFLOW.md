# Experiment Environment & LLM Role Workflow (Entropy-Aware KV Cache)

Last updated: 2026-01-15
Project: Entropy-Aware Dynamic KV Cache Recall Analysis

This document describes the **execution environment**, **storage policy**, and the **3-LLM collaboration workflow** used for our research on LLM inference optimization.
It is intended to be the single source of truth for how we run experiments and coordinate work across models.

---

## 1) Goals and Scope

### 1.1 Primary Goal
Validate the hypothesis that **High Token Entropy** during reasoning serves as a reliable trigger for **KV Cache Recall**.
- Observe the correlation between model uncertainty (Entropy) and cache hit rates (Recall).
- Demonstrate that static policies (H2O) fail to retain critical information during high-entropy reasoning steps.

### 1.2 What we optimize for
- **Logical Correctness:** Simulating eviction policies accurately without degrading the Oracle (Full Cache) baseline.
- **Metric Validity:** Ensuring Entropy and Recall are calculated consistently across different prompt lengths.
- **Reproducibility:** Experiments must be runnable on **Colab (A100)** with identical dependencies.

### 1.3 What we do *not* optimize for
- Actual wall-clock latency (at this stage). We focus on *simulation metrics* (Recall Rate).
- Multi-GPU distributed inference.

---

## 2) Execution Environment

### 2.1 Primary Runtime
- **Google Colab Pro+** is the primary execution environment.
- Target hardware: **Single NVIDIA A100 (40GB)**.
  - *Note:* L4 (24GB) is insufficient for LongBench context + FP16 model loading.

### 2.2 Software Stack
- **Framework:** PyTorch (`torch>=2.1.2`), Hugging Face Transformers.
- **Optimization:** **Flash Attention 2** (`flash-attn>=2.5.0`) is mandatory for memory efficiency.
- **Precision:** `torch.bfloat16` (Preferred) or `torch.float16`.

### 2.3 Model & Data
- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Dataset:** `THUDM/LongBench` (Subset: `hotpotqa`, Split: `test`)
- **Truncation:** Inputs truncated to **16k tokens** to ensure safety within A100 memory.

---

## 3) Storage & Versioning Policy (Critical)

We split workflows into two tracks.

### A) Official Experiment (Main Track) — Git used
**Single source of truth is the GitHub repo.**
- Code (`src/`), Configuration (`requirements.txt`), and Specs (`PROJECT.md`) are version-controlled.
- Colab is used for execution: **git clone → pip install → python src/experiment.py**.
- **Results are NOT committed to git.**

**Repo Layout:**
- `src/experiment.py` (Main logic)
- `src/utils.py` (Helpers)
- `PROJECT.md` (Spec)
- `EXPERIMENT_WORKFLOW.md` (This file)

**Outputs:**
- Stored in Google Drive (mounted at `/content/drive`).
- Format: **JSONL** (JSON Lines) for step-by-step logging.
- Naming: `results/{dataset}_budget_{ratio}_{timestamp}.jsonl`

### B) Exploratory Analysis (Deep Study) — Git not used
- Analysis Notebooks (`analysis.ipynb`) live in Google Drive.
- Visualization and aggregate statistics are generated here.
- Goal: Insight generation, not software artifacts.

---

## 4) 3-LLM Collaboration (Fixed Roles)

We use three LLMs with fixed responsibilities.

### 4.1 GPT = Planner / Spec Owner (Me)
**Responsibilities**
- Own the roadmap and experimental design (`PROJECT.md`).
- Define metrics (Entropy, Recall) and simulation logic.
- Analyze results and propose next steps.
- Maintain this workflow document.

**Not allowed**
- Writing implementation code directly (delegates to CODEX).

### 4.2 CODEX = Implementer
**Responsibilities**
- Implement Python code (`src/*.py`) exactly as specified in `PROJECT.md`.
- Handle PyTorch/HF APIs and CUDA memory management details.
- Return changes as **complete file contents** or **diffs**.

**Not allowed**
- Changing the logical definition of metrics (e.g., changing how Recall is calculated) without SPEC updates.

### 4.3 GEMINI = Reviewer / Verifier
**Responsibilities**
- Review `PROJECT.md` vs. Implementation vs. Results.
- Validate logic (e.g., "Is the H2O simulation leaking future information?").
- Suggest optimizations (e.g., vectorizing recall calculation).
- Report issues: **Problem → Cause → Fix**.

---

## 5) Measurement Invariants (Global Rules)

These rules define how we measure success.

### 5.1 Shadow Cache Simulation (The Core Invariant)
To measure recall without destroying performance:
1.  **Oracle:** The model runs with **Full KV Cache**. Actual generation uses this.
2.  **Policy (Shadow):** We maintain a separate "virtual" cache state (e.g., `accumulated_attention_scores` for H2O).
3.  **No Eviction:** We do **NOT** physically remove tokens from GPU memory during observation steps.

### 5.2 Metric Definitions
- **Token Entropy ($H_t$):** Calculated from `softmax(logits)` of the generated token. $H = -\sum p \log p$.
- **Ground Truth ($M_{gold}$):** Top-K indices from the *current step's* actual attention map (averaged across heads/layers).
- **Prediction ($M_{pred}$):** Top-K indices from the *policy's* scoring metric (e.g., accumulated attention).
- **Recall:** $|M_{pred} \cap M_{gold}| / K$.

### 5.3 Indices & Padding
- **Padding Tokens:** Must be excluded from both $M_{gold}$ and $M_{pred}$.
- **Special Tokens:** `<|im_start|>`, `<|im_end|>` are treated as normal context unless specified otherwise.
- **Shift:** Ensure attention indices map correctly to token IDs (considering causal masking).

---

## 6) Current Baseline

### Target Experiment: `HotpotQA` Observation
- **Setup:** A100, Qwen-2.5-7B, BF16.
- **Budgets:** Monitor Recall for compression ratios of **10%, 20%, 50%** simultaneously.
- **Output:** Per-token JSON log containing `step`, `entropy`, `recall_10`, `recall_20`, `recall_50`.

---
## 7) Operating Procedure (Checklist)

### 7.1 Execution Loop
1. **Plan:** GPT updates `PROJECT.md` with new hypotheses or settings.
2. **Code:** CODEX updates `src/experiment.py`.
3. **Run (Colab):**
   Run the following commands in a Colab cell:
   
   git clone [https://github.com/entropy-aware-kv-recall/project.git](https://github.com/entropy-aware-kv-recall/project.git)
   pip install -r requirements.txt
   python src/experiment.py --output_dir /content/drive/MyDrive/exp_logs

4. **Log:** Verify JSONL files are being written to Drive (check file size growing).
5. **Review:** GEMINI analyzes the logs for anomalies (e.g., Recall always 0.0 or 1.0).

### 7.2 Merge Gate for Policy Changes
A new cache policy (e.g., switching from H2O to SnapKV) can be added to the simulation only if:
- The simulation logic is verified by GEMINI (O(N) complexity check).
- It does not significantly slow down the Oracle inference speed (must remain feasible on Colab).

---

## 8) Reproducibility Requirements

Each execution log (JSONL metadata) must include:
- Model Name & Path
- Dataset & Split
- Max Sequence Length
- Flash Attention Version
- Timestamp