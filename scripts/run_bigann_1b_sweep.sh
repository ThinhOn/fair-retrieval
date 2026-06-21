#!/bin/bash
# run_bigann_1b_sweep.sh — full 1B SIFT/BIGANN scalability sweep over ell.
#
# Assumes data/bigann_1b/{vectors.dat,attributes.dat,metadata.npz} and the
# query pkl already exist (built by prepare_bigann.py + generate_queries_billion.py).
#
# For each ell in {4,8,16,32}: build the disk-cached LSH index (mu=2, w=800),
# run 200 queries, record preprocessing time + on-disk index size, then
# evaluate (success%, search/post time, scanned; recall@k is NaN — no GT at 1B).
#
# Resumable: --use-cache skips partitions already on disk, so a re-run continues.
set -e

DATASET=bigann_1b
DATA_DIR="./data/${DATASET}"
K=5; M=3; FDIST=euclidean; TARGET=200
W=800; MU=2; C=2.0; DELTA=0.1
NWORKERS=${NWORKERS:-8}
CHUNK=${CHUNK:-1000000}

PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8

SUMMARY="summary_${DATASET}.csv"
SCALE="scalability_${DATASET}.csv"
echo "ell,preprocessing_s,index_disk_GB" > "$SCALE"

for ELL in 4 8 16 32; do
    echo "=================== ELL=${ELL} ==================="
    LOG="/tmp/${DATASET}_ell${ELL}.log"
    "$PY" code/main.py \
        --data-dir "$DATA_DIR" \
        --index l2lsh_cartesian \
        --c "$C" --w "$W" --ell "$ELL" --mu "$MU" \
        --m "$M" --fdist "$FDIST" --delta "$DELTA" \
        --solver ilp --seed 10 \
        --use-cache --n_workers "$NWORKERS" --chunk_size "$CHUNK" 2>&1 | tee "$LOG" \
        | grep -E "Preprocessing time|Memory cost|returned no candidates|Indexing partitions: 100"

    PREP=$(grep -oE "Preprocessing time: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    CACHE_DIR="${DATA_DIR}/lsh_cache_ell=${ELL}_mu=${MU}_w=${W}.0_c=${C}_seed=10"
    DISK_GB=$(du -sb "$CACHE_DIR" 2>/dev/null | awk '{printf "%.3f", $1/1e9}')
    echo "${ELL},${PREP},${DISK_GB}" >> "$SCALE"
    echo ">>> ell=${ELL} preprocessing=${PREP}s index_disk=${DISK_GB}GB"
done

echo "=================== EVALUATING ==================="
"$PY" evaluate_ranged.py \
    --results_dir "./outputs/${DATASET}/results_ranged_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}" \
    --queries_path "${DATA_DIR}/ranged_queries_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}.pkl" \
    --output "$SUMMARY"

echo "=================== SCALABILITY (build) ==================="
cat "$SCALE"
echo "SWEEP_1B_DONE"
