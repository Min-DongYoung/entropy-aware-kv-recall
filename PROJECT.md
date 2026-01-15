# Project: Entropy-Aware Dynamic KV Cache Recall Analysis

## 1. Overview
**Objective:** To validate the hypothesis that **High Token Entropy** during reasoning serves as a reliable trigger for determining when existing static KV cache policies (e.g., H2O) fail to retain critical information.

**Context:** Efficient LLM inference requires KV cache compression. However, static policies based on accumulated attention scores often evict tokens that become crucial only during complex reasoning steps. This project observes the correlation between model uncertainty (Entropy) and cache hit rates (Recall).

## 2. Hypothesis
1.  **Correlation:** There is a strong negative correlation between **Token Entropy** and **KV Cache Recall** (under static policies like H2O).
    * *Low Entropy:* Static policies successfully retain needed info.
    * *High Entropy:* Static policies fail to retain newly needed info (Recall drops).
2.  **Sensitivity:** This phenomenon is more pronounced at lower **KV Cache Budgets** (higher compression ratios).

## 3. Experimental Setup

### 3.1. Infrastructure
* **Environment:** Google Colab Pro+ (Single Node)
* **GPU:** NVIDIA A100-SXM4-40GB (Required for full context simulation)
* **Model:** `Qwen/Qwen2.5-7B-Instruct` (Flash Attention 2 enabled)

### 3.2. Dataset
* **Benchmark:** **LongBench** (Reasoning Subset)
* **Primary Target:** `HotpotQA` (Multi-hop Reasoning, Clear Supporting Facts)
* **Secondary Target:** `MuSiQue` (For robustness check)

## 4. Methodology (Observation Pipeline)

The experiment performs a **Full KV Cache Inference** (Oracle) but simulates eviction policies in parallel to measure metrics without actual degradation.

### Step 1: Forward Pass & Data Collection
For each decoding step $t$:
1.  **Compute Token Entropy ($H_t$):**
    $$H_t = -\sum P(y_t) \log P(y_t)$$
2.  **Identify Oracle Needs ($M_{gold}^{(t)}$):**
    * Extract Top-$K$ indices from the **current** attention map (averaged across heads/layers).
3.  **Simulate Baseline Policy ($M_{pred}^{(t)}$):**
    * Track cumulative attention scores for all past tokens.
    * Identify Top-$K$ indices based on **accumulated** scores (simulating H2O retention).

### Step 2: Metric Calculation
1.  **Recall Rate:**
    $$Recall_t = \frac{|M_{pred}^{(t)} \cap M_{gold}^{(t)}|}{K}$$
2.  **Entropy Gap (Optional but Recommended):**
    * Measure the predictive difference if $M_{gold}$ was missing vs. present (Simulation).

## 5. Variables & Parameters

### Independent Variables (Conditions)
1.  **Token Entropy ($H_t$):** Analyzed in bins (e.g., Low < 1.0, Medium, High > 2.5).
2.  **Cache Budget Ratio ($r$):**
    * Test Cases: **10%, 20%, 30%** of total sequence length.
    * *Rationale:* To find the compression level where the entropy-recall tradeoff is most critical.

### Dependent Variables (Metrics)
1.  **Average Recall Rate:** The primary success metric.
2.  **Pearson Correlation Coefficient:** Between $H_t$ and $Recall_t$.

## 6. Directory Structure

/content/workspace
├── data/                  # LongBench dataset cache
├── results/
│   ├── hotpotqa_budget_0.1/  # Results for 10% budget
│   │   ├── logs.json         # Per-step [Entropy, Recall] data
│   │   └── plots/            # Generated visualization
│   ├── hotpotqa_budget_0.2/
│   └── ...
├── src/
│   ├── experiment.py      # Main loop (Forward pass + Simulation)
│   ├── metrics.py         # Entropy & Recall calculation
│   └── visualize.py       # Scatter plot & Trend line drawing
└── PROJECT.md             # This file

## 7. Expected Outcome & Analysis Plan
We expect to generate a **Binned Line Plot**:
* **X-Axis:** Token Entropy (Binned)
* **Y-Axis:** Average Recall Rate
* **Lines:** Different Cache Budgets (10%, 20%, 30%)

**Success Criteria:**
The graph should show a **downward trend**, where Recall remains high (~0.8+) for low entropy bins but drops significantly (<0.5) for high entropy bins, confirming that **"Uncertainty implies a need for Cache Update."**