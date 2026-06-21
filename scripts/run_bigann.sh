#!/bin/bash
# run_bigann.sh — end-to-end BIGANN / SIFT1B fair-retrieval pipeline.
#
# Usage:
#   scripts/run_bigann.sh <N> <DATASET_DIR_NAME> [ELL]
#
# Examples:
#   scripts/run_bigann.sh 10000      bigann_10k   16     # debug subset
#   scripts/run_bigann.sh 1000000000 bigann       32     # full billion
#
# Steps: (1) convert bvecs.gz -> memmaps, (2) generate ranged queries +
# ground truth, (3) build LSH index + run query processing, (4) evaluate.
#
# Set PY to the interpreter that has numpy/scipy/pulp/pandas installed.
set -e

N=${1:-10000}
DATASET=${2:-bigann_10k}
ELL=${3:-16}
W=${4:-800}          # bucket width — MUST scale with data; 800 is right for SIFT

K=5
M=3
FDIST=euclidean
TARGET=200

PY=${PY:-"C:/Users/Administrator/AppData/Local/Programs/Python/Python311/python.exe"}
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8

DATA_DIR="./data/${DATASET}"
SRC="data/bigann_base.bvecs.gz"

echo "=== [1/4] Preparing ${N} vectors -> ${DATA_DIR} ==="
"$PY" data/bigann/prepare_bigann.py --n "$N" --out_dir "$DATA_DIR" --src "$SRC"

echo "=== [2/4] Generating ${TARGET} ranged queries (k=${K}, m=${M}) ==="
"$PY" generate_queries_ranged.py --dataset "$DATASET" --fdist "$FDIST" --k "$K" --m "$M" --target "$TARGET"

echo "=== [3/4] Building LSH index + running query processing (ell=${ELL}, w=${W}) ==="
"$PY" code/main.py \
    --data-dir "$DATA_DIR" \
    --index l2lsh_cartesian \
    --c 2.0 --w "$W" --ell "$ELL" --mu 2 \
    --m "$M" --fdist "$FDIST" --delta 0.1 \
    --solver ilp --seed 10

echo "=== [4/4] Evaluating recall@k / success% ==="
RESULTS_DIR="./outputs/${DATASET}/results_ranged_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}"
QUERIES="${DATA_DIR}/ranged_queries_k=${K}_m=${M}_fdist=${FDIST}_${TARGET}.pkl"
"$PY" evaluate_ranged.py \
    --results_dir "$RESULTS_DIR" \
    --queries_path "$QUERIES" \
    --output "summary_${DATASET}.csv"

echo "=== Done. Summary written to summary_${DATASET}.csv ==="
