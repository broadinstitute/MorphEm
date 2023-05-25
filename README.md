# MorphEm

MorphEm_70k v2.0 is ready to download at [here](https://caicedolab.s3.us-west-2.amazonaws.com/MorphEm/morphem_70k_2.0.zip).

## Install

Install the `morphem` package by running
```
pip install git+git@github.com:broadinstitute/MorphEm.git#egg=morphem
```
Alternatively, run the evaluation pipeline through command line script in the `notebook` folder.

## Run Evaluation 

Importing the `run_benchmark` function from `benchmark.py` and passing required fields to the function.
Example of using the package from command line:
```
python -c "from morphem.benchmark import run_benchmark; 
run_benchmark('datasets/morphem_70k_2.0', 'results', 
'datasets/morphem_70k_2.0/features', 'pretrained_resnet18_features.npy')"
```
The function requires the following input parameters:  

`root_dir` : path to data directory.  
`dest_dir` : directory to store results.  
`feature_dir` : directory where features are stored.  
`feature_file` : filename of features.  

Optional field include:  

`classifier` : Default is 'knn'. Choose from 'knn' and 'sgd'.  
`umap` : Default is False. Whether or not to produce UMAP for features. 
`knn_metric` : Default is 'l2'. Choose from 'cosine' and 'l2'.

