#!/bin/bash
# run_ivf_celeba_ranged.sh — generate RANGED CelebA queries for m in {1,3,4,5}
# (fast indexed ground truth) and run IVF (reusing the built cache) + evaluate.
# Consistent with the ranged m=2 run. Per-m output summary_ivf_celeba_ranged_m=M.csv.
set -e
PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.; export PYTHONIOENCODING=utf-8
K=5; FDIST=euclidean; TARGET=200
NPROBES=${NPROBES:-"1 4 8"}

for M in 1 3 4 5; do
    QF="data/celeba/ranged_queries_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}.pkl"
    if [ ! -f "$QF" ]; then
        echo ">>> generating ranged queries celeba m=${M}"
        "$PY" generate_queries_ranged.py --dataset celeba --fdist "$FDIST" \
            --k "$K" --m "$M" --target "$TARGET" 2>&1 | grep -E "Precheck|Total final"
    fi
    for NPROBE in $NPROBES; do
        echo "=================== celeba(ranged) m=${M} nprobe=${NPROBE} ==================="
        "$PY" code/main.py --data-dir ./data/celeba --index l2ivf_cartesian \
            --nlist 0 --nprobe "$NPROBE" --oversample 100 \
            --m "$M" --fdist "$FDIST" --solver ilp --seed 10 2>&1 \
            | grep -E "Preprocessing time|returned no candidates"
    done
    echo "--- eval celeba(ranged) m=${M} ---"
    "$PY" evaluate_ranged.py \
        --results_dir "./outputs/celeba/results_ranged_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}" \
        --queries_path "$QF" \
        --output "summary_ivf_celeba_ranged_m=${M}.csv"
done
echo "IVF_CELEBA_RANGED_DONE"
