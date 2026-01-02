# fair-retrieval

Vectors + metadata + query files are uploaded here:


The structure of data folders should be as follows:
<project_folder>/data/<dataset>

For example, if the project folder is `fair-retrieval`, we can access the folder of audio dataset as: `fair-retrieval/audio/*`

## Query generation
To generate custom queries, run bash script:
    `bash scripts/generate_queries.sh <dataset> <distance_function> <num_attributes>`

    <dataset>: dataset name
    <distance_function>: distance function used ("euclidean", "cosine")
    <num_attributes>: number of constraints in queries (must be less than the number of attributes in the dataset)


## To run the main experiments for all algorithms (LSH, graph-based methods):
    `bash scripts/run_all.sh <dataset> <distance_func> <num_attributes>`