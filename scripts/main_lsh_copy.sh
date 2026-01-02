#! /bin/bash

BASE_PATH="."
export PYTHONPATH=${BASE_PATH}

# index
INDEX=${1}
# data
DATASET=${2}
DATA_DIR="./data/${DATASET}"
M=${3}
FDIST=${4}

C=2.0
DELTA=0.1

if [[ "$M" == 2 ]]; then
    SOLVER="network_flow"
else 
    SOLVER="ilp"
fi


### for debugging
C=2.0
L=4
W=4.0
MU=2
echo "Debugging"
echo "${INDEX}, ${DATASET}, ${FDIST}, c = ${C}, L = ${L}, w = ${W}"
OPTS=""
#data
OPTS+=" --data-dir ${DATA_DIR}"
#index
OPTS+=" --index ${INDEX}"
OPTS+=" --c ${C}"
OPTS+=" --w ${W}"
OPTS+=" --mu ${MU}"
OPTS+=" --ell ${L}"
OPTS+=" --m ${M}"
OPTS+=" --fdist ${FDIST}"
OPTS+=" --delta ${DELTA}"
#solver
OPTS+=" --solver ${SOLVER}"
# runtime
OPTS+=" --seed 10"
CMD="python ${BASE_PATH}/code/main_copy.py ${OPTS}"
${CMD}