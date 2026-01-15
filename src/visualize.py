import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_records(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "step" or "entropy" in row:
                records.append(row)
    return pd.DataFrame(records)


def add_bins(df: pd.DataFrame, bin_size: float, num_bins: int) -> pd.DataFrame:
    entropy = df["entropy"].astype(float)
    if num_bins > 0:
        df["entropy_bin"] = pd.qcut(entropy, num_bins, duplicates="drop")
    else:
        min_val = float(entropy.min())
        max_val = float(entropy.max())
        bins = np.arange(min_val, max_val + bin_size, bin_size)
        if bins.size < 2:
            bins = np.array([min_val, max_val + bin_size])
        df["entropy_bin"] = pd.cut(entropy, bins=bins, include_lowest=True)
    df["entropy_center"] = df["entropy_bin"].apply(lambda x: float(x.mid))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot binned recall vs. entropy.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "plots"))
    parser.add_argument("--bin_size", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_records(args.input_path)
    if df.empty:
        raise ValueError("No step records found in the JSONL file.")

    recall_cols = [col for col in df.columns if col.startswith("recall_")]
    if not recall_cols:
        raise ValueError("No recall_* columns found in the JSONL file.")
    df = add_bins(df, args.bin_size, args.num_bins)
    grouped = df.groupby("entropy_bin", observed=True)[recall_cols].mean().reset_index()
    grouped["entropy_center"] = grouped["entropy_bin"].apply(lambda x: float(x.mid))

    line_plot_df = grouped.melt(
        id_vars=["entropy_center"],
        value_vars=recall_cols,
        var_name="budget",
        value_name="recall",
    )
    line_plot_df["budget"] = line_plot_df["budget"].str.replace(
        "recall_", "budget_", regex=False
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=line_plot_df, x="entropy_center", y="recall", hue="budget", marker="o")
    plt.xlabel("Entropy (binned)")
    plt.ylabel("Mean Recall")
    plt.title("Binned Recall vs. Entropy")
    plt.tight_layout()

    output_path = os.path.join(args.output_dir, "binned_recall_vs_entropy.png")
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    boxen_df = df.melt(
        id_vars=["entropy_bin"],
        value_vars=recall_cols,
        var_name="budget",
        value_name="recall",
    )
    boxen_df["budget"] = boxen_df["budget"].str.replace("recall_", "budget_", regex=False)
    boxen_df["entropy_bin_label"] = boxen_df["entropy_bin"].astype(str)
    if hasattr(df["entropy_bin"].dtype, "categories"):
        bin_order = [str(cat) for cat in df["entropy_bin"].cat.categories]
    else:
        bin_order = [str(cat) for cat in sorted(df["entropy_bin"].unique(), key=lambda x: x.mid)]

    plt.figure(figsize=(10, 5))
    sns.boxenplot(
        data=boxen_df,
        x="entropy_bin_label",
        y="recall",
        hue="budget",
        order=bin_order,
    )
    plt.xlabel("Entropy (binned)")
    plt.ylabel("Recall")
    plt.title("Binned Recall vs. Entropy (Boxen)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    boxen_path = os.path.join(args.output_dir, "binned_recall_vs_entropy_boxen.png")
    plt.savefig(boxen_path, dpi=200)
    print(f"Saved plot to {boxen_path}")


if __name__ == "__main__":
    main()
