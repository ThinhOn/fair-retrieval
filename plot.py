import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load and combine all methods
# ------------------------------------------------------------
files = {
    "./outputs/fairface/LSH_Cartesian.xlsx":         "LSH Cartesian",
    # "LSH_Single.xlsx":            "LSH Single",
    # "BruteForce_Cartesian.xlsx":  "Brute Force per-partition",
}

dfs = []
for fname, method_name in files.items():
    try:
        df_tmp = pd.read_excel(fname)
    except FileNotFoundError:
        print(f"[WARN] File not found: {fname} – skipping")
        continue
    df_tmp["method"] = method_name
    dfs.append(df_tmp)

if not dfs:
    raise RuntimeError("No input files were loaded. Check filenames/paths.")

df = pd.concat(dfs, ignore_index=True)

# Derived metrics: average times per query
df["avg_search_time_per_query_s"]   = df["total_search_time_1000q_s"]   / 1000.0
df["avg_postproc_time_per_query_s"] = df["total_postproc_time_1000q_s"] / 1000.0

methods = sorted(df["method"].unique())
print("Methods found:", methods)

# ------------------------------------------------------------
# 2. Helper: automatically find best sweep for a given parameter
#    (where other params are constant and this param varies)
#    and the sweep is present in ALL methods
# ------------------------------------------------------------
def find_best_sweep(df, vary_param, methods):
    """
    Returns: (fixed_params_dict, sorted_list_of_vary_values)
    Example for vary_param='c':
      -> ({'r': 2.0, 'w': 1.0, 'delta': 0.1}, [1.0, 1.5, ..., 4.0])
    """
    all_params = ["c", "r", "w", "delta"]
    other_params = [p for p in all_params if p != vary_param]

    best_key = None
    best_vals = None

    # Group by all hyperparameters except the one we want to vary
    for key, sub in df.groupby(other_params):
        # key is either scalar or tuple; normalize to tuple
        if not isinstance(key, tuple):
            key = (key,)

        common_vals = None

        # For each method, collect the values of vary_param available
        for m in methods:
            sub_m = sub[sub["method"] == m]
            vals = sorted(sub_m[vary_param].unique().tolist())
            # Need at least 2 points per method to have a sweep
            if len(vals) < 2:
                common_vals = None
                break
            if common_vals is None:
                common_vals = set(vals)
            else:
                common_vals &= set(vals)

        # If some method does not have enough points, or intersection small, skip
        if common_vals is None or len(common_vals) < 2:
            continue

        # Keep the sweep with the largest common set of values
        if best_vals is None or len(common_vals) > len(best_vals):
            best_vals = sorted(common_vals)
            best_key = key

    if best_key is None:
        return None, None

    fixed_params = dict(zip(other_params, best_key))
    return fixed_params, best_vals

# ------------------------------------------------------------
# 3. Generic plotting function for one sweep
# ------------------------------------------------------------
def plot_sweep(df, vary_param, fixed_params, vary_values, title_prefix):
    """
    df: merged dataframe
    vary_param: 'c', 'r', or 'w'
    fixed_params: dict of constant params for this sweep
    vary_values: sorted list of parameter values to plot
    """
    if fixed_params is None or vary_values is None:
        print(f"[WARN] No sweep found for {vary_param}")
        return

    print(f"\n{title_prefix}")
    print(f"  Fixed hyperparameters: {fixed_params}")
    print(f"  {vary_param} values: {vary_values}")

    # Filter to the chosen sweep
    mask = pd.Series(True, index=df.index)
    for p, v in fixed_params.items():
        mask &= (df[p] == v)
    mask &= df[vary_param].isin(vary_values)

    d = df[mask].copy().sort_values(by=[vary_param, "method"])

    if d.empty:
        print(f"[WARN] No data after filtering for {vary_param}-sweep.")
        return

    metrics = [
        ("avg_error_rate_percent",        "Average Error Rate (%)",          "Error rate (%)"),
        ("total_correct_answers",         "Total Correct Answers",           "# correct"),
        ("avg_recall_k",                  "Average Recall@k",                "Recall@k"),
        ("avg_candidates_scanned",        "Avg # Candidates Scanned",        "# candidates"),
        ("time_build_indexes_s",          "Time for Building Indexes",       "Time (s)"),
        ("avg_search_time_per_query_s",   "Avg Search Time / Query",         "Time (s)"),
        ("avg_postproc_time_per_query_s", "Avg Post-processing Time / Query","Time (s)"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes = axes.ravel()

    for i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        for m in methods:
            sub_m = d[d["method"] == m].sort_values(by=vary_param)
            if sub_m.empty:
                continue
            ax.plot(sub_m[vary_param], sub_m[col], marker="o", label=m)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i >= len(metrics) - 1:
            ax.set_xlabel(vary_param)

    # Hide unused 8th subplot
    if len(metrics) < len(axes):
        axes[-1].axis("off")

    # One global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{title_prefix} – fixed: {fixed_params}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])

    plt.savefig(f"{vary_param}.png")
    plt.close()

# ------------------------------------------------------------
# 4. Find sweeps and plot
# ------------------------------------------------------------

# a) Sweep over c: keep (r, w, delta) constant
fixed_c, c_vals = find_best_sweep(df, "c", methods)
plot_sweep(df, "c", fixed_c, c_vals, "Varying c")

# b) Sweep over r: keep (c, w, delta) constant
fixed_r, r_vals = find_best_sweep(df, "r", methods)
plot_sweep(df, "r", fixed_r, r_vals, "Varying r")

# c) Sweep over w: keep (c, r, delta) constant
fixed_w, w_vals = find_best_sweep(df, "w", methods)
plot_sweep(df, "w", fixed_w, w_vals, "Varying w")
