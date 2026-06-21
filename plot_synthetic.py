"""
plot_synthetic.py
-----------------
Scalability + quality plots for the billion-scale SYNTHETIC sweep (d=32, clustered).
Linear x-axis throughout. Reads scalability_synthetic_1b.csv + summary_synthetic_1b.csv.
  (a) preprocessing time + index storage vs ell
  (b) query search time + post-processing vs ell
  (c) quality vs ell: success% (feasibility) + DAF (distance approx factor)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import re

scal = pd.read_csv("scalability_synthetic_1b.csv").sort_values("ell")
q = pd.read_csv("summary_synthetic_1b.csv")
q["ell"] = q["params"].apply(lambda s: int(re.search(r"ell=(\d+)", s).group(1)))
qm = q.sort_values("ell").set_index("ell")

ells     = scal["ell"].tolist()
build_h  = (scal["preprocessing_s"] / 3600.0).tolist()
disk_gb  = scal["index_disk_GB"].tolist()
search_s = (qm["avg_search_ms"] / 1000.0).reindex(ells).tolist()
post_ms  = qm["avg_post_ms"].reindex(ells).tolist()
success  = qm["success_%"].reindex(ells).tolist()
daf      = qm["DAF"].reindex(ells).tolist()

def _xfmt(ax):
    ax.set_xticks(ells); ax.set_xticklabels(ells)   # linear x-axis

# ── (a) build + storage ──────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
c1, c2 = "tab:blue", "tab:red"
ax1.plot(ells, build_h, "o-", color=c1); ax1.set_xlabel(r"$\ell$ (hash tables)")
ax1.set_ylabel("Preprocessing time (h)", color=c1); ax1.tick_params(axis="y", labelcolor=c1)
_xfmt(ax1)
ax2 = ax1.twinx(); ax2.plot(ells, disk_gb, "s--", color=c2)
ax2.set_ylabel("Index storage (GB)", color=c2); ax2.tick_params(axis="y", labelcolor=c2)
ax1.set_title("Synthetic 1B (d=32): preprocessing & storage")
fig.tight_layout(); fig.savefig("plot_synth_build_storage.png", dpi=200); fig.savefig("plot_synth_build_storage.pdf")
print("saved plot_synth_build_storage.png/.pdf")

# ── (b) query time ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))
c1, c2 = "tab:blue", "tab:green"
l1, = ax.plot(ells, search_s, "o-", color=c1)
ax.set_xlabel(r"$\ell$ (hash tables)"); ax.set_ylabel("Search time / query (s)", color=c1)
ax.tick_params(axis="y", labelcolor=c1); _xfmt(ax); ax.set_ylim(0, max(search_s)*1.08)
axp = ax.twinx(); l2, = axp.plot(ells, post_ms, "^--", color=c2)
axp.set_ylabel("Post-processing / query (ms)", color=c2); axp.tick_params(axis="y", labelcolor=c2)
axp.set_ylim(0, max(post_ms)*2.0)
ax.set_title("Synthetic 1B (d=32): query time vs $\\ell$")
ax.legend([l1, l2], ["Search (retrieval)", "Post-processing (ILP)"], loc="upper left")
fig.tight_layout(); fig.savefig("plot_synth_query.png", dpi=200); fig.savefig("plot_synth_query.pdf")
print("saved plot_synth_query.png/.pdf")

# ── (c) quality: success% + DAF ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))
c1, c2 = "tab:purple", "tab:orange"
l1, = ax.plot(ells, success, "o-", color=c1)
ax.set_xlabel(r"$\ell$ (hash tables)"); ax.set_ylabel("Successful queries (%)", color=c1)
ax.tick_params(axis="y", labelcolor=c1); _xfmt(ax); ax.set_ylim(0, 105)
axd = ax.twinx(); l2, = axd.plot(ells, daf, "s--", color=c2)
axd.set_ylabel("DAF (distance approx. factor)", color=c2); axd.tick_params(axis="y", labelcolor=c2)
axd.axhline(1.0, color=c2, lw=0.6, ls=":")
ax.set_title("Synthetic 1B (d=32): feasibility & quality vs $\\ell$")
ax.legend([l1, l2], ["Success %", "DAF (→ 1 = optimal)"], loc="center right")
fig.tight_layout(); fig.savefig("plot_synth_quality.png", dpi=200); fig.savefig("plot_synth_quality.pdf")
print("saved plot_synth_quality.png/.pdf")

print("\nell | build(h) | disk(GB) | success% | DAF | search(s) | scanned")
for e in ells:
    r = qm.loc[e]; s = scal[scal.ell == e].iloc[0]
    print(f"{e:3d} | {s.preprocessing_s/3600:7.2f} | {s.index_disk_GB:7.1f} | "
          f"{r['success_%']:7.1f} | {r['DAF']:.3f} | {r.avg_search_ms/1000:8.1f} | {r.avg_scanned:7.0f}")
