#!/usr/bin/env python3
"""Generate figures (PDF) and LaTeX table fragments from the results of Assignment 2.
Usage: python3 make_assets.py <results_dir> <out_dir>"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

R = Path(sys.argv[1]); O = Path(sys.argv[2]); (O/"figs").mkdir(exist_ok=True, parents=True); (O/"tables").mkdir(exist_ok=True, parents=True)
raw = pd.read_csv(R/"raw_measurements.csv")
raw["t"] = raw.time_ns/1e9
raw["label"] = raw.apply(lambda r: r.method + (f"(bs={int(r.block_size)})" if pd.notna(r.block_size) else ""), axis=1)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150})
STY = {"ijk_baseline": ("#444444", "o", "-"), "ikj_reorder": ("#0072B2", "s", "-"),
       "blocked_ikj(bs=32)": ("#E69F00", "^", "--"), "blocked_ikj(bs=64)": ("#009E73", "v", "--"), "blocked_ikj(bs=128)": ("#CC79A7", "D", "--")}
NAME = {"ijk_baseline": "ijk (baseline)", "ikj_reorder": "ikj (reordered)", "blocked_ikj(bs=32)": "blocked ikj, bs=32",
        "blocked_ikj(bs=64)": "blocked ikj, bs=64", "blocked_ikj(bs=128)": "blocked ikj, bs=128"}

def stat(g):
    q1, med, q3 = g.t.quantile([.25, .5, .75]); return pd.Series({"med": med, "q1": q1, "q3": q3, "iqr": q3-q1, "mn": g.t.min(), "mx": g.t.max()})

# ---------------- Part A
A = raw[(raw.experiment=="dense_opt")&(raw.phase=="multiply")]
sA = A.groupby(["label","n"]).apply(stat).reset_index()
sA["gflops"] = 2*sA.n**3/sA.med/1e9
sA.to_csv(O/"dense_stats.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lab,(c,m,ls) in STY.items():
    g = sA[sA.label==lab].sort_values("n")
    ax[0].errorbar(g.n, g.med, yerr=[g.med-g.q1, g.q3-g.med], color=c, marker=m, ms=3.5, ls=ls, lw=1.1, capsize=1.5, label=NAME[lab])
    ax[1].plot(g.n, g.gflops, color=c, marker=m, ms=3.5, ls=ls, lw=1.1, label=NAME[lab])
ax[0].set(xscale="log", yscale="log", xlabel="matrix size $n$", ylabel="median time per multiplication (s)")
ax[1].set(xscale="log", xlabel="matrix size $n$", ylabel=r"throughput $2n^3/t$ (GFLOP/s)", ylim=(0, 5))
ax[0].set_title("(a) Runtime (bars: IQR)", fontsize=9); ax[1].set_title("(b) Throughput", fontsize=9)
for a in ax: a.set_xticks([100,200,400,800,1600,3200]); a.set_xticklabels([100,200,400,800,1600,3200])
ax[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(O/"figs/fig_dense.pdf"); plt.close(fig)

# tables
def fmt_ms(x):
    ms = x*1e3
    return f"{ms:.2f}" if ms < 10 else (f"{ms:.1f}" if ms < 100 else f"{ms:,.0f}".replace(",", "\\,"))
order = ["ijk_baseline","ikj_reorder","blocked_ikj(bs=32)","blocked_ikj(bs=64)","blocked_ikj(bs=128)"]
hdr = ["ijk","ikj","blk 32","blk 64","blk 128"]
rows = []
for n in sorted(sA.n.unique()):
    cells = []
    for lab in order:
        g = sA[(sA.label==lab)&(sA.n==n)]
        cells.append("--" if g.empty else f"{fmt_ms(g.med.iloc[0])} ({g.iqr.iloc[0]*1e3:.2g})")
    rows.append(f"{n} & " + " & ".join(cells) + r" \\")
(O/"tables/dense_time.tex").write_text("\n".join(rows))

base = sA[sA.label=="ijk_baseline"].set_index("n").med; ikj = sA[sA.label=="ikj_reorder"].set_index("n").med
rows = []
for n in sorted(sA.n.unique()):
    c = [f"{base[n]/ikj[n]:.2f}" if n in base.index else "--"]
    for lab in order[2:]:
        g = sA[(sA.label==lab)&(sA.n==n)].med.iloc[0]
        c.append(f"{base[n]/g:.2f}" if n in base.index else "--")
    for lab in order[2:]:
        g = sA[(sA.label==lab)&(sA.n==n)].med.iloc[0]
        c.append(f"{ikj[n]/g:.2f}")
    rows.append(f"{n} & " + " & ".join(c) + r" \\")
(O/"tables/dense_speedup.tex").write_text("\n".join(rows))

# dense memory
M = raw[(raw.experiment=="dense_opt")&(raw.phase=="memory_live_heap")&(raw.method=="ikj_reorder")].sort_values("n")
rows = [f"{int(r.n)} & {r.est_storage_bytes/1e6:.1f} & {r.heap_used_bytes/1e6:.1f} & {r.peak_rss_kb*1024/1e6:.0f} \\\\" for r in M.itertuples()]
(O/"tables/dense_mem.tex").write_text("\n".join(rows))

# ---------------- Part B
B = raw[(raw.experiment=="sparse_vs_dense")]
def med(df, **kw):
    q = df
    for k,v in kw.items(): q = q[q[k]==v]
    return q.t.median(), q.t.quantile(.75)-q.t.quantile(.25)
ver = B[(B.phase=="verify")&(B.method=="csr_gustavson")].set_index(["n","sparsity_pct"])
memcsr = B[(B.phase=="memory_live_heap")&(B.method=="csr_gustavson")].set_index(["n","sparsity_pct"])
memikj = B[(B.phase=="memory_live_heap")&(B.method=="ikj_reorder")].set_index(["n","sparsity_pct"])
recs = []
for n in sorted(B.n.unique()):
    for sp in sorted(B.sparsity_pct.unique()):
        s = B[(B.n==n)&(B.sparsity_pct==sp)]
        r = dict(n=n, sp=sp, dens=100-sp)
        r["nnzA"] = ver.loc[(n,sp),"nnz_a"]; r["nnzB"] = ver.loc[(n,sp),"nnz_b"]; r["nnzC"] = ver.loc[(n,sp),"nnz_c"]
        r["ijk"], r["ijk_iqr"] = med(s[(s.method=="ijk_baseline")&(s.phase=="multiply")], n=n) if len(s[(s.method=="ijk_baseline")&(s.phase=="multiply")]) else (np.nan, np.nan)
        r["ikj"], r["ikj_iqr"] = med(s[(s.method=="ikj_reorder")&(s.phase=="multiply")])
        r["cmul"], r["cmul_iqr"] = med(s[(s.method=="csr_gustavson")&(s.phase=="multiply")])
        r["conv"], r["conv_iqr"] = med(s[(s.method=="csr_gustavson")&(s.phase=="convert_dense_to_csr")])
        r["e2e"], r["e2e_iqr"] = med(s[(s.method=="csr_gustavson")&(s.phase=="convert_plus_multiply")])
        r["csr_bytes"] = memcsr.loc[(n,sp),"est_storage_bytes"]; r["rss_csr"] = memcsr.loc[(n,sp),"peak_rss_kb"]
        r["rss_ikj"] = memikj.loc[(n,sp),"peak_rss_kb"]
        da, db = r["nnzA"]/n**2, r["nnzB"]/n**2
        r["predC_dens"] = 100*(1-(1-da*db)**n); r["measC_dens"] = 100*r["nnzC"]/n**2
        recs.append(r)
S = pd.DataFrame(recs); S.to_csv(O/"sparse_stats.csv", index=False)

def ft(x):
    if np.isnan(x): return "--"
    return f"{x:.4f}" if x < 0.01 else (f"{x:.3f}" if x < 1 else f"{x:.2f}")
def sp_fmt(x):
    return f"{x:.1f}" if x < 100 else f"{x:,.0f}"
rows = []
for r in S.itertuples():
    rows.append(f"{r.n} & {r.sp:g} & {int(r.nnzA):,} & {int(r.nnzC):,} & {r.measC_dens:.2f} & {ft(r.ikj)} & {ft(r.cmul)} & {ft(r.conv)} & {ft(r.e2e)} & {sp_fmt(r.ikj/r.cmul)} & {sp_fmt(r.ikj/r.e2e)} \\\\".replace(",", "\\,"))
    if r.sp == 99.9 and r.n != 4000: rows.append(r"\midrule")
(O/"tables/sparse_main.tex").write_text("\n".join(rows))

rows = []
for r in S.itertuples():
    dense_b = 3*8*r.n**2
    rows.append(f"{r.n} & {r.sp:g} & {r.csr_bytes/1e6:.2f} & {dense_b/1e6:.0f} & {r.csr_bytes/dense_b*100:.1f} & {r.rss_csr*1024/1e6:.0f} & {r.rss_ikj*1024/1e6:.0f} \\\\")
    if r.sp == 99.9 and r.n != 4000: rows.append(r"\midrule")
(O/"tables/sparse_mem.tex").write_text("\n".join(rows))

rows = [f"{r.n} & {r.sp:g} & {r.measC_dens:.2f} & {r.predC_dens:.2f} \\\\" for r in S.itertuples()]
(O/"tables/nnz_pred.tex").write_text("\n".join(rows))

# Fig sparse time
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=False)
for a, n in zip(ax, sorted(S.n.unique())):
    g = S[S.n==n].sort_values("dens")
    a.plot(g.dens, g.ijk, color="#444444", marker="o", ms=3.5, lw=1.1, label="dense ijk")
    a.plot(g.dens, g.ikj, color="#0072B2", marker="s", ms=3.5, lw=1.1, label="dense ikj")
    a.plot(g.dens, g.cmul, color="#D55E00", marker="^", ms=3.5, lw=1.1, label="CSR multiply")
    a.plot(g.dens, g.e2e, color="#56B4E9", marker="v", ms=3.5, lw=1.1, ls="--", label="CSR convert+multiply")
    a.set(xscale="log", yscale="log", xlabel="density of A and B (%)", title=f"$n={n}$"); a.set_xticks([0.1,1,5,10]); a.set_xticklabels(["0.1","1","5","10"])
    a.invert_xaxis()
ax[0].set_ylabel("median time (s)"); ax[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(O/"figs/fig_sparse_time.pdf"); plt.close(fig)

# Fig nnz(C)
fig, ax = plt.subplots(figsize=(3.6, 2.8))
cols = {1000: "#0072B2", 2000: "#D55E00", 4000: "#009E73"}
for n in sorted(S.n.unique()):
    g = S[S.n==n].sort_values("dens")
    ax.plot(g.dens, g.predC_dens, color=cols[n], ls=":", lw=1)
    ax.plot(g.dens, g.measC_dens, color=cols[n], marker="o", ms=4, ls="none", label=f"measured, $n={n}$")
ax.plot([], [], color="k", ls=":", lw=1, label=r"$1-(1-d_Ad_B)^n$")
ax.set(xscale="log", yscale="log", xlabel="density of A and B (%)", ylabel="density of $C$ (%)"); ax.set_xticks([0.1,1,5,10]); ax.set_xticklabels(["0.1","1","5","10"]); ax.invert_xaxis()
ax.legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(O/"figs/fig_nnz.pdf"); plt.close(fig)

# Fig memory
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for n in sorted(S.n.unique()):
    g = S[S.n==n].sort_values("dens")
    ax.plot(g.dens, g.csr_bytes/1e6, color=cols[n], marker="o", ms=4, label=f"CSR A+B+C, $n={n}$")
    ax.axhline(3*8*n**2/1e6, color=cols[n], ls=":", lw=1)
ax.plot([], [], color="k", ls=":", lw=1, label="dense A+B+C")
ax.set(xscale="log", yscale="log", xlabel="density of A and B (%)", ylabel="array storage (MB)"); ax.set_xticks([0.1,1,5,10]); ax.set_xticklabels(["0.1","1","5","10"]); ax.invert_xaxis()
ax.legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(O/"figs/fig_mem.pdf"); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 40)
print(S[["n","sp","nnzA","nnzC","measC_dens","predC_dens","ikj","cmul","conv","e2e"]].round(4).to_string())
print(sA.pivot(index="n", columns="label", values="gflops").round(2))
print("max IQR/median dense:", (sA.iqr/sA.med).max(), sA.loc[(sA.iqr/sA.med).idxmax(),["label","n"]].tolist())
print("ijk 1000 90% iqr", S[(S.n==1000)&(S.sp==90)][["ijk","ijk_iqr"]].values)

# ---- per-operation cost (estimate) and growth exponents
rows = []
for r in S.itertuples():
    prods = r.nnzA*r.nnzB/r.n            # expected number of scalar products of the Gustavson kernel (uniform nonzeros)
    rows.append(f"{r.n} & {r.sp:g} & {prods/1e6:.3f} & {r.cmul/prods*1e9:.2f} & {r.ikj/r.n**3*1e9:.2f} & {(r.cmul/prods)/(r.ikj/r.n**3):.1f} \\\\".replace(",", "\\,"))
    if r.sp == 99.9 and r.n != 4000: rows.append(r"\midrule")
(O/"tables/sparse_cost.tex").write_text("\n".join(rows))
print("\nGrowth exponents (OLS log t vs log n):")
for lab in order:
    g = sA[sA.label==lab]
    for lo in (100, 1600):
        h = g[g.n>=lo]
        if len(h) >= 3:
            k = np.polyfit(np.log(h.n), np.log(h.med), 1)[0]; print(f"  {lab:22s} n>={lo}: {k:.2f}")
print(S[["n","sp","ikj","ijk"]].pivot(index="n", columns="sp", values="ikj").round(3))
print(S[["n","sp","ikj","ijk"]].pivot(index="n", columns="sp", values="ijk").round(3))
