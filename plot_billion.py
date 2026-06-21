"""
plot_billion.py
---------------
Paper-style scalability plots for the full 1B SIFT/BIGANN sweep (Figure 11):
  (a) preprocessing time + index storage vs ell (dual y-axis)
  (b) query processing time (search + post-processing) vs ell

Reads scalability_bigann_1b.csv (ell, preprocessing_s, index_disk_GB) and
summary_bigann_1b.csv (per-config query metrics). Saves PNG + PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import re

scal = pd.read_csv("scalability_bigann_1b.csv").sort_values("ell")

q = pd.read_csv("summary_bigann_1b.csv")
q["ell"] = q["params"].apply(lambda s: int(re.search(r"ell=(\d+)", s).group(1)))
q = q.sort_values("ell")
qm = q.set_index("ell")

ells          = scal["ell"].tolist()
build_h       = (scal["preprocessing_s"] / 3600.0).tolist()
disk_gb       = scal["index_disk_GB"].tolist()
search_s      = (qm["avg_search_ms"] / 1000.0).reindex(ells).tolist()
post_ms       = qm["avg_post_ms"].reindex(ells).tolist()

# ── (a) build time + storage vs ell ──────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
c1, c2 = "tab:blue", "tab:red"
ax1.plot(ells, build_h, "o-", color=c1, label="Build time")
ax1.set_xlabel(r"$\ell$ (hash tables)")
ax1.set_ylabel("Preprocessing time (h)", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_xticks(ells); ax1.set_xticklabels(ells)   # linear x-axis

ax2 = ax1.twinx()
ax2.plot(ells, disk_gb, "s--", color=c2, label="Index storage")
ax2.set_ylabel("Index storage (GB)", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)

ax1.set_title("1B SIFT: preprocessing & storage vs $\\ell$")
fig.tight_layout()
fig.savefig("plot_1b_build_storage.png", dpi=200)
fig.savefig("plot_1b_build_storage.pdf")
print("saved plot_1b_build_storage.png/.pdf")

# ── (b) query time vs ell (linear axes; post-proc on twin axis in ms) ─────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))
c1, c2 = "tab:blue", "tab:green"
l1, = ax.plot(ells, search_s, "o-", color=c1, label="Search (retrieval)")
ax.set_xlabel(r"$\ell$ (hash tables)")
ax.set_ylabel("Search time / query (s)", color=c1)
ax.tick_params(axis="y", labelcolor=c1)
ax.set_xticks(ells); ax.set_xticklabels(ells)   # linear x-axis
ax.set_ylim(0, max(search_s) * 1.08)

axp = ax.twinx()
l2, = axp.plot(ells, post_ms, "^--", color=c2, label="Post-processing (ILP)")
axp.set_ylabel("Post-processing / query (ms)", color=c2)
axp.tick_params(axis="y", labelcolor=c2)
axp.set_ylim(0, max(post_ms) * 2.0)

ax.set_title("1B SIFT: query time vs $\\ell$  (100% feasible)")
ax.legend(handles=[l1, l2], loc="upper left")
fig.tight_layout()
fig.savefig("plot_1b_query.png", dpi=200)
fig.savefig("plot_1b_query.pdf")
print("saved plot_1b_query.png/.pdf")

# ── console summary ──────────────────────────────────────────────────────────
print("\nell | build(h) | disk(GB) | search(s) | post(ms) | scanned | success%")
for e in ells:
    r = qm.loc[e]
    print(f"{e:3d} | {scal[scal.ell==e].preprocessing_s.iloc[0]/3600:7.2f} | "
          f"{scal[scal.ell==e].index_disk_GB.iloc[0]:7.1f} | "
          f"{r.avg_search_ms/1000:8.1f} | {r.avg_post_ms:7.1f} | "
          f"{r.avg_scanned:7.0f} | {r['success_%']:.0f}")
