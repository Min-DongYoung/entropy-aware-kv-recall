# Entropy-Aware Dynamic KV Cache Recall Analysis

This project observes the correlation between High Token Entropy and KV Cache Recall failures under static policies (H2O simulation). It runs full-context (oracle) inference while simulating cache retention to measure recall without evicting tokens. Full experimental details are in `PROJECT.md`.

## Environment Setup
```bash
pip install -r requirements.txt
```
Designed for NVIDIA A100 (40GB) with Flash Attention 2.

## Usage Guide
Run the experiment:
```bash
python src/experiment.py --output_dir ./results --model_path "Qwen/Qwen2.5-7B-Instruct"
```

Visualize results:
```bash
python src/visualize.py --input_path ./results/your_log.jsonl
```

## Output Structure
- JSONL logs are saved under `results/`
- Plots are saved under `results/plots/`
