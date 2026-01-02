#! /bin/bash

METHODS=(
    sieve_cartesian
    # filter_diskann
    # brute_force_cartesian
)

DATASET=${1}
FDIST=${2-"euclidean"}
M_ATTR=${3}


## add LSH methods based on distance function
# if [[ "$FDIST" == "euclidean" ]]; then
#     METHODS+=(
#         l2lsh_cartesian
#         l2lsh_single
#         l2lsh_joint
#     )
# elif [[ "$FDIST" == "cosine" ]]; then
#     METHODS+=(
#         angular_lsh_cartesian
#         angular_lsh_single
#         angular_lsh_joint
#     )
# fi

for method in "${METHODS[@]}"; do
    ## determine main file to be called
    if [[ "$method" == *lsh* ]]; then
        MAIN="main_lsh.sh"
    elif [[ "$method" == *sieve* ]]; then
        MAIN="main_sieve_copy.sh"
    elif [[ "$method" == *disk* ]]; then
        MAIN="main_diskann.sh"
    elif [[ "$method" == *brute* ]]; then
        MAIN="main_bruteforce.sh"
    else
        echo "Unknown method: $method" >&2
        exit 1
    fi

    CMD="bash scripts/${MAIN} ${method} ${DATASET} ${M_ATTR} ${FDIST}"
    echo "Running: ${CMD}"
    ${CMD}
done