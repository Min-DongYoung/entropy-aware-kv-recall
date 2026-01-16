import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def load_step_records(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "step":
                records.append(row)
    if not records:
        raise ValueError("No step records found in the JSONL file.")
    return pd.DataFrame(records)


def detect_recall_cols(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col.startswith("recall_")]


def add_delta_columns(df: pd.DataFrame, recall_cols: list) -> list:
    delta_cols = []

    def maybe_add(col_a: str, col_b: str, name: str) -> None:
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a] - df[col_b]
            delta_cols.append(name)
        else:
            missing = [col for col in (col_a, col_b) if col not in df.columns]
            warnings.warn(
                f"Skipping {name}; missing columns: {', '.join(missing)}",
                RuntimeWarning,
            )

    maybe_add("recall_20", "recall_10", "delta_recall_20_10")
    maybe_add("recall_10", "recall_05", "delta_recall_10_05")
    maybe_add("recall_50", "recall_20", "delta_recall_50_20")

    return delta_cols


def add_entropy_bins(df: pd.DataFrame, num_bins: int, bin_size: float) -> pd.DataFrame:
    entropy = df["entropy"].astype(float)
    if num_bins > 0:
        df["entropy_bin"] = pd.qcut(entropy, num_bins, duplicates="drop")
    else:
        min_val = float(entropy.min())
        max_val = float(entropy.max())
        bins = np.arange(min_val, max_val + bin_size, bin_size)
        if bins.size < 2:
            bins = np.array([min_val, min_val + bin_size])
        df["entropy_bin"] = pd.cut(entropy, bins=bins, include_lowest=True)
    df["entropy_center"] = df["entropy_bin"].apply(lambda x: float(x.mid))
    return df


def compute_sample_lengths(df: pd.DataFrame) -> pd.DataFrame:
    sample_stats = (
        df.groupby("sample_id", observed=True)["step"]
        .agg(steps="count", min_step="min", max_step="max")
        .reset_index()
    )
    steps = sample_stats["steps"].to_numpy()
    summary = {
        "sample_id": "__summary__",
        "steps": np.nan,
        "min_step": np.nan,
        "max_step": np.nan,
        "mean_steps": float(np.mean(steps)) if steps.size else np.nan,
        "median_steps": float(np.median(steps)) if steps.size else np.nan,
        "std_steps": float(np.std(steps, ddof=1)) if steps.size > 1 else np.nan,
        "p90_steps": float(np.percentile(steps, 90)) if steps.size else np.nan,
        "p95_steps": float(np.percentile(steps, 95)) if steps.size else np.nan,
    }
    for col in ["mean_steps", "median_steps", "std_steps", "p90_steps", "p95_steps"]:
        sample_stats[col] = np.nan
    return pd.concat([sample_stats, pd.DataFrame([summary])], ignore_index=True)


def _bin_centers(df: pd.DataFrame) -> pd.Series:
    return df["entropy_bin"].apply(lambda x: float(x.mid))


def _add_metric_stats(grouped, metric_cols: list) -> pd.DataFrame:
    stats_map = {}
    for col in metric_cols:
        stats_map[f"{col}_mean"] = grouped[col].mean()
        stats_map[f"{col}_median"] = grouped[col].median()
        stats_map[f"{col}_std"] = grouped[col].std()
        stats_map[f"{col}_iqr"] = grouped[col].quantile(0.75) - grouped[col].quantile(0.25)
    return pd.DataFrame(stats_map)


def summarize_token_bins(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    grouped = df.groupby("entropy_bin", observed=True)
    counts = grouped.size().rename("count")
    stats_df = _add_metric_stats(grouped, metric_cols)
    result = pd.concat([counts, stats_df], axis=1).reset_index()
    result["entropy_center"] = _bin_centers(result)
    return result


def summarize_sample_bins(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    sample_bin_means = (
        df.groupby(["sample_id", "entropy_bin"], observed=True)[metric_cols]
        .mean()
        .reset_index()
    )
    grouped = sample_bin_means.groupby("entropy_bin", observed=True)
    counts = grouped.size().rename("count_samples")
    stats_df = _add_metric_stats(grouped, metric_cols)
    result = pd.concat([counts, stats_df], axis=1).reset_index()
    result["entropy_center"] = _bin_centers(result)
    return result


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 2:
        return {"spearman_r": np.nan, "spearman_p": np.nan}
    r, p = stats.spearmanr(x, y)
    return {"spearman_r": float(r), "spearman_p": float(p)}


def _safe_linregress(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "rvalue": np.nan,
            "pvalue": np.nan,
            "stderr": np.nan,
        }
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "rvalue": float(res.rvalue),
        "pvalue": float(res.pvalue),
        "stderr": float(res.stderr),
    }


def compute_trends(summary_df: pd.DataFrame, recall_cols: list) -> dict:
    trends = {}
    x_all = summary_df["entropy_center"].to_numpy()
    for col in recall_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std" if f"{col}_std" in summary_df.columns else f"{col}_iqr"

        x = x_all
        y = summary_df[mean_col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]

        trend = {}
        mean_spearman = _safe_spearman(x_valid, y_valid)
        mean_regress = _safe_linregress(x_valid, y_valid)
        trend.update(
            {
                "mean_spearman_r": mean_spearman["spearman_r"],
                "mean_spearman_p": mean_spearman["spearman_p"],
                "mean_slope": mean_regress["slope"],
                "mean_intercept": mean_regress["intercept"],
                "mean_rvalue": mean_regress["rvalue"],
                "mean_pvalue": mean_regress["pvalue"],
                "mean_stderr": mean_regress["stderr"],
            }
        )

        if std_col in summary_df.columns:
            v = summary_df[std_col].to_numpy()
            vmask = np.isfinite(x) & np.isfinite(v)
            v_valid = v[vmask]
            x_vol = x[vmask]
            vol_spearman = _safe_spearman(x_vol, v_valid)
            trend.update(
                {
                    "volatility_metric": std_col,
                    "volatility_spearman_r": vol_spearman["spearman_r"],
                    "volatility_spearman_p": vol_spearman["spearman_p"],
                }
            )
        else:
            trend["volatility_metric"] = None
            trend["volatility_spearman_r"] = np.nan
            trend["volatility_spearman_p"] = np.nan

        required_keys = [
            "mean_spearman_r",
            "mean_spearman_p",
            "mean_slope",
            "mean_intercept",
            "mean_rvalue",
            "mean_pvalue",
            "mean_stderr",
            "volatility_metric",
            "volatility_spearman_r",
            "volatility_spearman_p",
        ]
        missing = [key for key in required_keys if key not in trend]
        if missing:
            raise RuntimeError(f"Trend summary missing keys for {col}: {missing}")

        trends[col] = trend
    return trends


def pick_primary_recall(recall_cols: list) -> str:
    if not recall_cols:
        return ""

    def key_fn(col: str) -> float:
        match = re.findall(r"[\d.]+", col)
        if not match:
            return float("inf")
        try:
            return float(match[0])
        except ValueError:
            return float("inf")

    return sorted(recall_cols, key=key_fn)[0]


def _safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    df_pair = pd.concat([series_a, series_b], axis=1).dropna()
    if df_pair.shape[0] < 2:
        return np.nan
    return float(df_pair.iloc[:, 0].corr(df_pair.iloc[:, 1], method=method))


def compute_time_lag_summary(df: pd.DataFrame) -> dict:
    required_cols = {"sample_id", "entropy", "attn_entropy"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        warnings.warn(
            f"Skipping time-lag analysis; missing columns: {', '.join(missing)}",
            RuntimeWarning,
        )
        return {"skipped": True, "missing_columns": missing}

    lag_df = df.copy()
    lag_df["next_attn_entropy"] = (
        lag_df.groupby("sample_id", observed=True)["attn_entropy"].shift(-1)
    )
    lag_df["next_token_entropy"] = (
        lag_df.groupby("sample_id", observed=True)["entropy"].shift(-1)
    )

    lag_df = lag_df.dropna(subset=["next_attn_entropy", "next_token_entropy", "entropy", "attn_entropy"])

    concurrent_pearson = _safe_corr(lag_df["entropy"], lag_df["attn_entropy"], "pearson")
    concurrent_spearman = _safe_corr(lag_df["entropy"], lag_df["attn_entropy"], "spearman")
    forward_pearson = _safe_corr(lag_df["entropy"], lag_df["next_attn_entropy"], "pearson")
    forward_spearman = _safe_corr(lag_df["entropy"], lag_df["next_attn_entropy"], "spearman")
    backward_pearson = _safe_corr(lag_df["attn_entropy"], lag_df["next_token_entropy"], "pearson")
    backward_spearman = _safe_corr(lag_df["attn_entropy"], lag_df["next_token_entropy"], "spearman")

    high_thresh = float(lag_df["entropy"].quantile(0.9)) if not lag_df.empty else np.nan
    low_thresh = float(lag_df["entropy"].quantile(0.1)) if not lag_df.empty else np.nan
    high_mean = (
        float(lag_df.loc[lag_df["entropy"] > high_thresh, "next_attn_entropy"].mean())
        if np.isfinite(high_thresh)
        else np.nan
    )
    low_mean = (
        float(lag_df.loc[lag_df["entropy"] < low_thresh, "next_attn_entropy"].mean())
        if np.isfinite(low_thresh)
        else np.nan
    )

    return {
        "skipped": False,
        "counts": {
            "rows_total": int(df.shape[0]),
            "rows_used": int(lag_df.shape[0]),
        },
        "correlations": {
            "concurrent": {
                "pearson": concurrent_pearson,
                "spearman": concurrent_spearman,
            },
            "forward": {
                "pearson": forward_pearson,
                "spearman": forward_spearman,
            },
            "backward": {
                "pearson": backward_pearson,
                "spearman": backward_spearman,
            },
        },
        "conditional_next_attn_entropy": {
            "high_entropy_threshold": high_thresh,
            "low_entropy_threshold": low_thresh,
            "mean_next_attn_entropy_high": high_mean,
            "mean_next_attn_entropy_low": low_mean,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JSONL logs from experiment.py.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--bin_size", type=float, default=0.1)
    parser.add_argument(
        "--aggregate_mode", type=str, choices=["token", "sample", "both"], default="both"
    )
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--primary_recall", type=str, default="recall_10")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_step_records(args.input_path)
    recall_cols = detect_recall_cols(df)
    if not recall_cols:
        raise ValueError("No recall_* columns detected in the JSONL file.")
    delta_cols = add_delta_columns(df, recall_cols)
    metric_cols = recall_cols + delta_cols

    df = add_entropy_bins(df, args.num_bins, args.bin_size)

    sample_lengths = compute_sample_lengths(df)
    sample_lengths_path = os.path.join(args.output_dir, "sample_lengths.csv")
    sample_lengths.to_csv(sample_lengths_path, index=False)

    trend_summary = {}
    if args.aggregate_mode in ("token", "both"):
        token_summary = summarize_token_bins(df, metric_cols)
        token_path = os.path.join(args.output_dir, "binned_summary_token.csv")
        token_summary.to_csv(token_path, index=False)
        trend_summary["token"] = compute_trends(token_summary, metric_cols)

    if args.aggregate_mode in ("sample", "both"):
        sample_summary = summarize_sample_bins(df, metric_cols)
        sample_path = os.path.join(args.output_dir, "binned_summary_sample.csv")
        sample_summary.to_csv(sample_path, index=False)
        trend_summary["sample"] = compute_trends(sample_summary, metric_cols)

    trend_path = os.path.join(args.output_dir, "trend_summary.json")
    with open(trend_path, "w", encoding="utf-8") as handle:
        json.dump(trend_summary, handle, indent=2)

    if args.primary_recall in df.columns:
        primary_recall = args.primary_recall
    else:
        primary_recall = pick_primary_recall(recall_cols)

    if primary_recall:
        exemplars = df.sort_values(
            by=["entropy", primary_recall], ascending=[False, True]
        ).head(args.top_n)
    else:
        exemplars = df.sort_values(by=["entropy"], ascending=[False]).head(args.top_n)
    exemplar_cols = ["sample_id", "step", "entropy"] + recall_cols
    exemplars_path = os.path.join(args.output_dir, "exemplars.csv")
    exemplars[exemplar_cols].to_csv(exemplars_path, index=False)

    time_lag_summary = compute_time_lag_summary(df)
    time_lag_path = os.path.join(args.output_dir, "time_lag_summary.json")
    with open(time_lag_path, "w", encoding="utf-8") as handle:
        json.dump(time_lag_summary, handle, indent=2)

    print("Analysis complete.")
    print(f"Records: {len(df)}")
    print(f"Samples: {df['sample_id'].nunique()}")
    print(f"Recall columns: {', '.join(recall_cols)}")
    print(f"Sample lengths: {sample_lengths_path}")
    if args.aggregate_mode in ("token", "both"):
        print(f"Token summary: {token_path}")
    if args.aggregate_mode in ("sample", "both"):
        print(f"Sample summary: {sample_path}")
    print(f"Trend summary: {trend_path}")
    print(f"Exemplars: {exemplars_path}")
    print(f"Time-lag summary: {time_lag_path}")
    if not time_lag_summary.get("skipped"):
        corr = time_lag_summary["correlations"]
        cond = time_lag_summary["conditional_next_attn_entropy"]
        print("Time-lag correlations:")
        print(
            "  concurrent (entropy vs attn_entropy): "
            f"pearson={corr['concurrent']['pearson']:.4f}, "
            f"spearman={corr['concurrent']['spearman']:.4f}"
        )
        print(
            "  forward (entropy -> next_attn_entropy): "
            f"pearson={corr['forward']['pearson']:.4f}, "
            f"spearman={corr['forward']['spearman']:.4f}"
        )
        print(
            "  backward (attn_entropy -> next_token_entropy): "
            f"pearson={corr['backward']['pearson']:.4f}, "
            f"spearman={corr['backward']['spearman']:.4f}"
        )
        print(
            "Next attn_entropy conditioned on entropy(t): "
            f"high>{cond['high_entropy_threshold']:.4f} mean={cond['mean_next_attn_entropy_high']:.4f}, "
            f"low<{cond['low_entropy_threshold']:.4f} mean={cond['mean_next_attn_entropy_low']:.4f}"
        )


if __name__ == "__main__":
    main()
