## Install

Install the `evaluate_pipeline` package by running
```
pip install git+git@github.com:broadinstitute/MorphEm.git#egg=evaluate_pipeline
```

## Run Evaluation
Run the evaluation pipeline by importing the `run_benchmark` function from `run_benchmark.py` and passing required fields to the function.
Example of using the package from command line:
```
python -c "from evaluate_pipeline.run_benchmark import run_benchmark; 
run_benchmark('datasets/morphem_70k_2.0', 'results_delete', 
'datasets/morphem_70k_2.0/features', 'pretrained_resnet18_features.npy')"
```
The function requires the following input parameters:  

`root_dir` : path to data directory.  
`dest_dir` : directory to store results.  
`feature_dir` : directory where features are stored.  
`feature_file` : filename of features.  

Optional parameters include:  
`classifier` : choose from 'knn' and 'sgd'. Default is 'knn'.
`umap` : whether or not to produce UMAP for features. Default is False.
`use_gpu` : use GPU or CPU for KNN classification. Default is True (use GPU).
