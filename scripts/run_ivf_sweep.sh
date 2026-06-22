#!/bin/bash
# run_ivf_sweep.sh <dataset> [m] [fdist]  — per-partition IVF (Design A) nprobe sweep.
# Build happens once (use_cache, nlist auto); nprobe is query-time so subsequent
# values reuse the cached indexes. Set NPROBES env to control the sweep (default
# a low config "1 4 8"). Queries with ground truth => recall@k + DAF reported.
set -e
DATASET=${1:-bigann_1m}
M=${2:-3}
FDIST=${3:-euclidean}
NPROBES=${NPROBES:-"1 4 8"}
K=5; TARGET=200
PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.; export PYTHONIOENCODING=utf-8

for NPROBE in $NPROBES; do
    echo "=================== ${DATASET} m=${M} nprobe=${NPROBE} ==================="
    "$PY" code/main.py --data-dir "./data/${DATASET}" --index l2ivf_cartesian \
        --nlist 0 --nprobe "$NPROBE" --oversample 100 \
        --m "$M" --fdist "$FDIST" --solver ilp --seed 10 2>&1 \
        | grep -E "Preprocessing time|returned no candidates"
done
echo "=================== EVALUATING ==================="
"$PY" evaluate_ranged.py \
    --results_dir "./outputs/${DATASET}/results_ranged_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}" \
    --queries_path "./data/${DATASET}/ranged_queries_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}.pkl" \
    --output "summary_ivf_${DATASET}.csv"
echo "IVF_SWEEP_DONE ${DATASET}"
