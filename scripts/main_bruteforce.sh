#! /bin/bash

BASE_PATH="."
export PYTHONPATH=${BASE_PATH}

# index
INDEX=${1-"brute_force_cartesian"}
# data
DATASET=${2-"fairface"}
DATA_DIR="./data/${DATASET}"

M_ATTR=${3}
FDIST=${4-"euclidean"}

OPTS=""
#data
OPTS+=" --data-dir ${DATA_DIR}"
#index
OPTS+=" --index ${INDEX}"
OPTS+=" --m ${M_ATTR}"
OPTS+=" --fdist ${FDIST}"
#solver
OPTS+=" --solver ilp"
# runtime
OPTS+=" --seed 10"
CMD="python ${BASE_PATH}/code/main.py ${OPTS}"

${CMD}