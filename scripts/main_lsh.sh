#! /bin/bash

BASE_PATH="."
export PYTHONPATH=${BASE_PATH}

# index
INDEX=${1-"l2lsh_cartesian"}
# data
DATASET=${2-"audio"}
DATA_DIR="./data/${DATASET}"
M=${3}
FDIST=${4-"euclidean"}

C=2.0
DELTA=0.1


### vary W
L=16
MU=2
for W in 4.0 6.0 8.0 10.0; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, c = ${C}, L = ${L}, w = ${W}", mu = ${MU}
    OPTS=""
    #data
    OPTS+=" --data-dir ${DATA_DIR}"
    #index
    OPTS+=" --index ${INDEX}"
    OPTS+=" --c ${C}"
    # OPTS+=" --r ${R}"
    OPTS+=" --w ${W}"
    OPTS+=" --ell ${L}"
    OPTS+=" --mu ${MU}"
    OPTS+=" --m ${M}"
    OPTS+=" --fdist ${FDIST}"
    OPTS+=" --delta ${DELTA}"
    #solver
    OPTS+=" --solver ilp"
    # runtime
    OPTS+=" --seed 10"

    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done


### vary ell
W=4.0
MU=2
for L in 8 16 32 64 128; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, c = ${C}, L = ${L}, w = ${W}", mu = ${MU}
    OPTS=""
    #data
    OPTS+=" --data-dir ${DATA_DIR}"
    #index
    OPTS+=" --index ${INDEX}"
    OPTS+=" --c ${C}"
    # OPTS+=" --r ${R}"
    OPTS+=" --w ${W}"
    OPTS+=" --ell ${L}"
    OPTS+=" --mu ${MU}"
    OPTS+=" --m ${M}"
    OPTS+=" --fdist ${FDIST}"
    OPTS+=" --delta ${DELTA}"
    #solver
    OPTS+=" --solver ilp"
    # runtime
    OPTS+=" --seed 10"

    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done


### vary concat length
W=4.0
L=16
for MU in 1 2 4 8; do
    echo ""
    echo "${INDEX}, ${DATASET}, ${FDIST}, c = ${C}, L = ${L}, w = ${W}", mu = ${MU}
    OPTS=""
    #data
    OPTS+=" --data-dir ${DATA_DIR}"
    #index
    OPTS+=" --index ${INDEX}"
    OPTS+=" --c ${C}"
    # OPTS+=" --r ${R}"
    OPTS+=" --w ${W}"
    OPTS+=" --ell ${L}"
    OPTS+=" --mu ${MU}"
    OPTS+=" --m ${M}"
    OPTS+=" --fdist ${FDIST}"
    OPTS+=" --delta ${DELTA}"
    #solver
    OPTS+=" --solver ilp"
    # runtime
    OPTS+=" --seed 10"
    CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
    ${CMD}
done


### for debugging
# C=2.0
# L=4
# W=4.0
# echo "Debugging"
# echo "${INDEX}, ${DATASET}, ${FDIST}, c = ${C}, L = ${L}, w = ${W}"
# OPTS=""
# #data
# OPTS+=" --data-dir ${DATA_DIR}"
# #index
# OPTS+=" --index ${INDEX}"
# OPTS+=" --c ${C}"
# OPTS+=" --w ${W}"
# OPTS+=" --ell ${L}"
# OPTS+=" --m ${M}"
# OPTS+=" --fdist ${FDIST}"
# OPTS+=" --delta ${DELTA}"
# #solver
# OPTS+=" --solver ilp"
# # runtime
# OPTS+=" --seed 10"
# CMD="python ${BASE_PATH}/code/main.py ${OPTS}"
# ${CMD}