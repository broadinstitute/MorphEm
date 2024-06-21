# CHAMMI

The CHAMMI dataset is available for download on [Zenodo](https://zenodo.org/record/7988357).

We released model checkpoints and the training scripts to further aid reproducibility. 
Our code and checkpoints are now publicly available at [here](https://github.com/chaudatascience/channel_adaptive_models).

## Disclosure

These instructions are made for conda environments, but the `requirements.txt` file is given for other users

### Notice

We also use 'pip' commands since both 'pip' and 'conda' are generally compatible and can be used in one 
environment 


## Do Before Running Code

Download the dataset/images by clicking the Zenodo link above (downloading the dataset takes around 2 hours 
so do this as soon as possible).
Then, run the "Feature Extraction" notebook located in the `notebooks` section to obtain the features

## Checking and installing correct python version

Check python version by running
```
python --version
```

If your python version is not 3.8 or 3.9, install python with
`conda install python=3.8` or `conda install python=3.9`

### Recommended Version:

```
conda install python=3.9
```

#### Notice

If you choose to install version 3.8, be aware that there will be some warning messages that will
display after running the code from the "Run Evaluation" section below. To avoid these warning
messages, install python version 3.9


## Install

Install the `morphem` package by running within the repo folder
```
pip install -e .
```

Next, install faiss-gpu version 1.7.3 with this command
```
conda install -c pytorch faiss-gpu==1.7.3
```

We use conda because faiss-gpu cannot be installed using pip


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

### Notice

Make sure to put the complete directory for `root_dir`, `dest_dir`, and `feature_dir` in order
for the code to run


Optional field include:  

`classifier` : Default is 'knn'. Choose from 'knn' and 'sgd'.  
`umap` : Default is False. Whether or not to produce UMAP for features. 
`knn_metric` : Default is 'l2'. Choose from 'cosine' and 'l2'.

## Example Images from each Dataset with Labels
![alt text](https://github.com/broadinstitute/MorphEm/blob/main/example_image.png)
