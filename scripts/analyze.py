#!/usr/bin/env python3
"""Summarise raw_measurements.csv: medians, IQR, speedups, tables and plots.
Usage: python3 scripts/analyze.py [results_dir]
Writes summary_*.csv and figures *.png into the same directory."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
df = pd.read_csv(d / "raw_measurements.csv")
df["time_s"] = df["time_ns"] / 1e9
df["label"] = df.apply(lambda r: r["method"] + (f"(bs={int(r['block_size'])})" if pd.notna(r["block_size"]) else ""), axis=1)

def stats(g):
    q1, med, q3 = g["time_s"].quantile([.25, .5, .75])
    return pd.Series({"reps": len(g), "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                      "min_s": g["time_s"].min(), "max_s": g["time_s"].max(),
                      "nnz_c": g["nnz_c"].median(), "any_budget_exceeded": (g["status"] != "OK").any()})

def summarise(frame, keys):
    return frame.groupby(keys, dropna=False).apply(stats).reset_index()

# ---------- Part A
A = df[(df.experiment == "dense_opt") & (df.phase == "multiply") & df.time_ns.notna()]
if len(A):
    sA = summarise(A, ["label", "n"])
    base = sA[sA.label == "ijk_baseline"].set_index("n")["median_s"]
    sA["speedup_vs_baseline"] = sA.apply(lambda r: base.get(r["n"], float("nan")) / r["median_s"], axis=1)
    sA["gflops"] = sA.apply(lambda r: 2 * r["n"] ** 3 / r["median_s"] / 1e9, axis=1)
    sA.to_csv(d / "summary_dense.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, g in sA.groupby("label"):
        g = g.sort_values("n")
        ax[0].errorbar(g.n, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], marker="o", capsize=2, label=lab)
        ax[1].plot(g.n, g.speedup_vs_baseline, marker="o", label=lab)
    ax[0].set(xscale="log", yscale="log", xlabel="matrix size n", ylabel="median time per multiplication (s)", title="Dense runtime (bars: IQR)")
    ax[1].set(xscale="log", xlabel="matrix size n", ylabel="speedup vs ijk_baseline", title="Speedup of optimisations")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    for a in ax: a.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(d / "fig_dense_runtime_speedup.png", dpi=150)

# ---------- Part B
B = df[(df.experiment == "sparse_vs_dense") & df.time_ns.notna() &
       df.phase.isin(["multiply", "convert_dense_to_csr", "convert_plus_multiply"])]
if len(B):
    B = B.assign(series=B.apply(lambda r: (r["method"] + " [" + r["phase"] + "]") if r["method"].startswith("csr") else r["method"] + " [dense]", axis=1))
    sB = summarise(B, ["series", "n", "sparsity_pct"])
    ref = sB[sB.series == "ikj_reorder [dense]"].set_index(["n", "sparsity_pct"])["median_s"]
    sB["speedup_vs_dense_ikj"] = sB.apply(lambda r: ref.get((r["n"], r["sparsity_pct"]), float("nan")) / r["median_s"], axis=1)
    sB.to_csv(d / "summary_sparse.csv", index=False)
    ns = sorted(sB.n.unique())
    fig, axs = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4), squeeze=False)
    for ax, n in zip(axs[0], ns):
        for s, g in sB[sB.n == n].groupby("series"):
            g = g.sort_values("sparsity_pct")
            ax.plot(g.sparsity_pct, g.median_s, marker="o", label=s)
        ax.set(yscale="log", xlabel="sparsity (% zeros)", ylabel="median time (s)", title=f"n = {n}")
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(d / "fig_sparse_vs_dense.png", dpi=150)

    # nonzero growth: nnz(C) relative to inputs
    N = df[(df.experiment == "sparse_vs_dense") & (df.phase == "verify") & (df.method.str.startswith("csr"))]
    if len(N):
        N = N.assign(nnz_c_over_nnz_a=N.nnz_c / N.nnz_a, nnz_c_density_pct=100 * N.nnz_c / N.n ** 2)
        N[["n", "sparsity_pct", "nnz_a", "nnz_b", "nnz_c", "nnz_c_over_nnz_a", "nnz_c_density_pct", "verified"]].to_csv(d / "summary_nnz.csv", index=False)

# ---------- Memory
M = df[df.phase.isin(["memory_live_heap", "memory_csr_delta"])]
if len(M):
    M[["experiment", "label", "phase", "n", "sparsity_pct", "est_storage_bytes", "heap_used_bytes", "peak_rss_kb"]].to_csv(d / "summary_memory.csv", index=False)

# ---------- Status / limits
S = df[(df.phase.isin(["multiply", "warmup"])) & (df.status != "OK")]
S[["experiment", "label", "n", "sparsity_pct", "phase", "status", "time_s"]].drop_duplicates().to_csv(d / "summary_limits.csv", index=False)
bad = df[df.verified == "FAIL"]
print(f"Wrote summaries and figures to {d}. Verification failures: {len(bad)}")
