# CHAMMI

The CHAMMI dataset is available for download at [here](https://zenodo.org/record/7988357).
The code we used for training and evaluating models are publicly available at [here](https://github.com/chaudatascience/channel_adaptive_models).
## Install

Install the `morphem` package by running within the repo folder
```
pip install -e .
```
Alternatively, run the evaluation pipeline through command line script in the `notebook` folder.

## Run Evaluation 

Importing the `run_benchmark` function from `benchmark.py` and passing required fields to the function.
Example of using the package from command line:
```
python -c "from morphem.benchmark import run_benchmark; 
run_benchmark('datasets/CHAMMI', 'results', 
'datasets/CHAMMI/features', 'pretrained_resnet18_features.npy')"
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

