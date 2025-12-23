#! /bin/bash

BASE_PATH="."
export PYTHONPATH=${BASE_PATH}

# index
INDEX=${1-"filter_diskann"}
# data
DATASET=${2-"audio"}
DATA_DIR="./data/${DATASET}"

M_ATTR=${3}
FDIST=${4}

MIN_BUCKET_SIZE=1

FILTERING_MULT=${5-4}


### vary EF_CONSTRUCTION
M=8
EF_SEARCH=4
for EF_CONSTRUCTION in 8 16 32; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, efc = ${EF_CONSTRUCTION}, M = ${M}, efs = ${EF_SEARCH}, mult = ${FILTERING_MULT}"
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
    OPTS+=" --filtering-multiplier ${FILTERING_MULT}"
    #solver
    OPTS+=" --solver ilp"
    # runtime
    OPTS+=" --seed 10"
    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done


### vary M
EF_CONSTRUCTION=8
EF_SEARCH=4
for M in 8 16 32; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, efc = ${EF_CONSTRUCTION}, M = ${M}, efs = ${EF_SEARCH}, mult = ${FILTERING_MULT}"
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
    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done


### vary EF_SEARCH
M=8
EF_CONSTRUCTION=8
for EF_SEARCH in 4 8 16; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, efc = ${EF_CONSTRUCTION}, M = ${M}, efs = ${EF_SEARCH}, mult = ${FILTERING_MULT}"
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
    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done