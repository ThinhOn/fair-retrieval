#! /bin/bash

BASE_PATH="."
export PYTHONPATH=${BASE_PATH}

# index
INDEX=${1-"sieve_cartesian"}
# data
DATASET=${2-"fairface"}
DATA_DIR="./data/${DATASET}"

M_ATTR=${3}
FDIST=${4}

MIN_BUCKET_SIZE=1


### vary EF_CONSTRUCTION
EF_CONSTRUCTION=128
M=128
EF_SEARCH=64
echo ""
echo "${INDEX}, ${DATASET}, ${FDIST}, efc = ${EF_CONSTRUCTION}, M = ${M}, efs = ${EF_SEARCH}"
OPTS=""
#data
OPTS+=" --data-dir ${DATA_DIR}"
#index
OPTS+=" --index ${INDEX}"
OPTS+=" --ef-construction ${EF_CONSTRUCTION}"
OPTS+=" --M ${M}"
OPTS+=" --ef-search ${EF_SEARCH}"
OPTS+=" --m ${M_ATTR}"
OPTS+=" --fdist ${FDIST}"
OPTS+=" --min-bucket-size ${MIN_BUCKET_SIZE}"
#solver
OPTS+=" --solver ilp"
# runtime
OPTS+=" --seed 10"
CMD="python ${BASE_PATH}/code/main_copy.py ${OPTS}"
${CMD}