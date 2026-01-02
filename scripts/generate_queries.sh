#! /bin/bash

DATASET=${1-"audio"}
FDIST=${2-"euclidean"}
M_ATTR=${3-3}

python generate_queries.py --dataset=${DATASET} --m=${M_ATTR} --fdist=${FDIST} --k=5
python generate_queries.py --dataset=${DATASET} --m=${M_ATTR} --fdist=${FDIST} --k=10
python generate_queries.py --dataset=${DATASET} --m=${M_ATTR} --fdist=${FDIST} --k=15
python generate_queries.py --dataset=${DATASET} --m=${M_ATTR} --fdist=${FDIST} --k=20
python generate_queries.py --dataset=${DATASET} --m=${M_ATTR} --fdist=${FDIST} --k=25

# for m in 1 2 3 4 5; do
#     echo "Generating queries for dataset=${DATASET}, m=${m}, fdist=${FDIST}, k=20"
#     python generate_queries_vary_m.py --dataset=${DATASET} --m=${m} --fdist=${FDIST} --k=20 --target=1000
# done