# CHAMMI

The CHAMMI dataset is available for download on [Zenado](https://zenodo.org/record/7988357).

We released model checkpoints and the training scripts to further aid reproducibility. 
Our code and checkpoints are now publicly available at [here](https://github.com/chaudatascience/channel_adaptive_models).


### Note

**These instructions are made for conda environments, but the `requirements.txt` file is given for other users.**

**We also use 'pip' commands since both 'pip' and 'conda' are generally compatible and can be used in one 
environment.**


## Checking and installing the correct python version

Check python version by running
```
python --version
```

We used python versions 3.8 and 3.9 to test the code, but we recommend using version 3.9
```
conda install python=3.9
```

**If you choose to install version 3.8, be aware that there will be some warning messages that will 
display after running the code from the "Run Evaluation" section below.**


## Install

Install the `morphem` package by running within the repo folder:
```
pip install -e .
```

Next, install faiss-gpu version 1.7.3 with this command:
```
conda install -c pytorch faiss-gpu==1.7.3
```

We use conda because faiss-gpu cannot be installed using pip.


## Getting Features

Run the "Feature_Extraction.py" file located in the `morphem` directory with the appropriate arguments. Before running the code, make sure to run
```
nvidia-smi
```
to check which gpu is available/"more free". Put that gpu number in that corresponding argument's place when running the "Feature_Extraction.py" file (the gpu number is the last argument to be passed).

### Note

Make sure to be inside the `morphem` directory when running the "Feature_Extraction.py" file.


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

**Make sure to put the complete directory for **`root_dir`, `dest_dir`,** and **`feature_dir`** in order
for the code to run**


Optional field include:  

`classifier` : Default is 'knn'. Choose from 'knn' and 'sgd'.  
`umap` : Default is False. Whether or not to produce UMAP for features. 
`knn_metric` : Default is 'l2'. Choose from 'cosine' and 'l2'.

## Example Images from each Dataset with Labels
![alt text](https://github.com/broadinstitute/MorphEm/blob/main/example_image.png)
