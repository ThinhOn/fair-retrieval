#!/bin/bash
# run_ivf_celeba_m.sh — CelebA IVF across m using the pre-existing exact-count
# queries_k=5_m=N^5_*.pkl files (they already carry ground truth). The IVF index
# is identical across m (partitioning is by all 5 attrs), so the cache built for
# the m=2 run is reused — these runs only query + evaluate.
set -e
PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.; export PYTHONIOENCODING=utf-8
FDIST=euclidean; K=5
NPROBES=${NPROBES:-"1 4 8"}

for M in 1 3 4 5; do
    QF="data/celeba/queries_k=${K}_m=${M}^5_fdist=${FDIST}_200.pkl"
    TAG="k=${K}_m=${M}^5_fdist=${FDIST}_200"
    for NPROBE in $NPROBES; do
        echo "=================== celeba m=${M} nprobe=${NPROBE} ==================="
        "$PY" code/main.py --data-dir ./data/celeba --index l2ivf_cartesian \
            --nlist 0 --nprobe "$NPROBE" --oversample 100 \
            --m "$M" --fdist "$FDIST" --solver ilp --seed 10 \
            --query-file "$QF" --query-tag "$TAG" 2>&1 \
            | grep -E "Preprocessing time|returned no candidates"
    done
    echo "--- eval celeba m=${M} ---"
    "$PY" evaluate_ranged.py \
        --results_dir "./outputs/celeba/results_ranged_${TAG}" \
        --queries_path "$QF" \
        --output "summary_ivf_celeba_m=${M}.csv"
done
echo "IVF_CELEBA_M_DONE"
