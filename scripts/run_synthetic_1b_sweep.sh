#!/bin/bash
# run_synthetic_1b_sweep.sh — full ell sweep on the billion-scale SYNTHETIC data
# (data/synthetic: n=1e9, d=32, well-separated Gaussian clusters, 36 partitions).
# Same protocol as the SIFT/BIGANN 1B sweep but w=4.0 (unit-scale data) and the
# existing 200-query file (which carries ground truth → recall@k + DAF reported).
set -e

DATASET=synthetic
DATA_DIR="./data/${DATASET}"
K=5; M=3; FDIST=euclidean; TARGET=200
W=4.0; MU=2; C=2.0; DELTA=0.1

PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8

SUMMARY="summary_synthetic_1b.csv"
SCALE="scalability_synthetic_1b.csv"
echo "ell,preprocessing_s,index_disk_GB" > "$SCALE"

for ELL in 4 8 16 32 64; do
    echo "=================== ELL=${ELL} ==================="
    LOG="/tmp/synthetic_1b_ell${ELL}.log"
    "$PY" code/main.py \
        --data-dir "$DATA_DIR" \
        --index l2lsh_cartesian \
        --c "$C" --w "$W" --ell "$ELL" --mu "$MU" \
        --m "$M" --fdist "$FDIST" --delta "$DELTA" \
        --solver ilp --seed 10 \
        --use-cache --n_workers 4 --chunk_size 500000 2>&1 | tee "$LOG" \
        | grep -E "Preprocessing time|returned no candidates|Indexing partitions: 100"

    PREP=$(grep -oE "Preprocessing time: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    CACHE_DIR="${DATA_DIR}/lsh_cache_ell=${ELL}_mu=${MU}_w=${W}_c=${C}_seed=10"
    DISK_GB=$(du -sb "$CACHE_DIR" 2>/dev/null | awk '{printf "%.3f", $1/1e9}')
    echo "${ELL},${PREP},${DISK_GB}" >> "$SCALE"
    echo ">>> ell=${ELL} preprocessing=${PREP}s index_disk=${DISK_GB}GB"
done

echo "=================== EVALUATING ==================="
"$PY" evaluate_ranged.py \
    --results_dir "./outputs/${DATASET}/results_ranged_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}" \
    --queries_path "${DATA_DIR}/ranged_queries_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}.pkl" \
    --output "$SUMMARY"
cat "$SCALE"
echo "SWEEP_SYNTH_DONE"
