"""
plot_combined.py
----------------
Combined SIFT-vs-synthetic billion-scale figure (2x2 panels), both datasets
overlaid as separate series. Linear x-axis throughout.
  (a) preprocessing time vs ell      (b) index storage vs ell
  (c) search time / query vs ell      (d) successful queries (%) vs ell
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import re

# publication style
plt.rcParams.update({
    "savefig.dpi": 300, "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 7,
})


def load(scal_csv, summ_csv):
    s = pd.read_csv(scal_csv).sort_values("ell")
    q = pd.read_csv(summ_csv)
    q["ell"] = q["params"].apply(lambda x: int(re.search(r"ell=(\d+)", x).group(1)))
    q = q.sort_values("ell").set_index("ell")
    ells = s["ell"].tolist()
    return {
        "ells":     ells,
        "build_h":  (s["preprocessing_s"] / 3600.0).tolist(),
        "disk_gb":  s["index_disk_GB"].tolist(),
        "search_s": (q["avg_search_ms"] / 1000.0).reindex(ells).tolist(),
        "success":  q["success_%"].reindex(ells).tolist(),
    }

sift  = load("scalability_bigann_1b.csv",    "summary_bigann_1b.csv")
synth = load("scalability_synthetic_1b.csv", "summary_synthetic_1b.csv")

ALL_ELLS = sorted(set(sift["ells"]) | set(synth["ells"]))
SIFT_KW  = dict(color="tab:red",  marker="o", ls="-",  label="SIFT/BIGANN (d=128)")
SYN_KW   = dict(color="tab:blue", marker="s", ls="--", label="Synthetic (d=32)")

def setx(ax):
    ax.set_xticks(ALL_ELLS); ax.set_xticklabels(ALL_ELLS); ax.set_xlabel(r"$\ell$ (hash tables)")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# (a) build time
ax = axes[0, 0]
ax.plot(sift["ells"], sift["build_h"], **SIFT_KW)
ax.plot(synth["ells"], synth["build_h"], **SYN_KW)
setx(ax); ax.set_ylabel("Preprocessing time (h)")
ax.set_title("(a) Build time vs $\\ell$"); ax.legend()

# (b) storage
ax = axes[0, 1]
ax.plot(sift["ells"], sift["disk_gb"], **SIFT_KW)
ax.plot(synth["ells"], synth["disk_gb"], **SYN_KW)
setx(ax); ax.set_ylabel("Index storage (GB)")
ax.set_title("(b) Index storage vs $\\ell$  (identical: 4 GB$\\times\\ell$)"); ax.legend()

# (c) search time (log y — datasets differ ~30x)
ax = axes[1, 0]
ax.plot(sift["ells"], sift["search_s"], **SIFT_KW)
ax.plot(synth["ells"], synth["search_s"], **SYN_KW)
setx(ax); ax.set_ylabel("Search time / query (s)"); ax.set_yscale("log")
ax.set_title("(c) Query search time vs $\\ell$"); ax.legend()

# (d) success %
ax = axes[1, 1]
ax.plot(sift["ells"], sift["success"], **SIFT_KW)
ax.plot(synth["ells"], synth["success"], **SYN_KW)
setx(ax); ax.set_ylabel("Successful queries (%)"); ax.set_ylim(0, 105)
ax.set_title("(d) Feasibility vs $\\ell$"); ax.legend(loc="lower right")

fig.suptitle("Billion-scale fair $k$-NN: SIFT/BIGANN vs Synthetic", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("plot_combined_1b.png", dpi=200)
fig.savefig("plot_combined_1b.pdf")
print("saved plot_combined_1b.png/.pdf")
