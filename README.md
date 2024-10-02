# CHAMMI

The CHAMMI dataset is available for download on [Zenodo](https://zenodo.org/record/7988357).

We released model checkpoints and the training scripts to further aid reproducibility. 
Our code and checkpoints are now publicly available at [here](https://github.com/chaudatascience/channel_adaptive_models).

The CHAMMI paper can be found [here](https://arxiv.org/pdf/2310.19224.pdf) and a short presentation of 
the project can be found [here](https://neurips.cc/virtual/2023/poster/73620).

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

Run the "feature_extraction.py" file located in the `morphem` directory with the appropriate arguments. Before running the code, make sure to run
```
nvidia-smi
```
to check which gpu is available/"more free". Put that gpu number in that corresponding argument's place when running the "Feature_Extraction.py" file (the gpu number is the second-to-last argument to be passed).

An example of running the "feature_extraction.py" file is shown below:
```
python feature_extraction.py --root_dir (root directory) --feat_dir (directory that stores features) --model (model type) --gpu 0 --batch_size 64
```

Replace the parenthesis with your directories and model type appropriately.


### Note

Make sure to be inside the `morphem` directory when running the "Feature_Extraction.py" file.
Also be sure to put a large batch size to minimize computing time, but also small enough that works for your gpu size.


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

## Expected Results from Each Dataset
### Macro Average Results Comparison
|             *Dataset*     :arrow_right:    | **Allen/WTC**  |                      |     **HPA**     |        |        |    **CP**    |        |        |        |
|----------------------|---------------|----------|---------|--------|--------|--------|--------|--------|--------|
| ***Model Type***   :arrow_down:|    Task 1     |    Task 2       |    Task 1      |  Task 2    | Task 3 | Task 1 | Task 2 | Task 3 | Task 4 |
| **ResNet**            |     0.54     |  0.48    |   0.52  |  0.33  |  0.21  |  0.63  |  0.33  |  0.27  |  0.09  |
| **ConvNext**          |      0.55     |  0.37    |   0.56  |  0.42  |  0.25  |  0.84  |  0.48  |  0.32  |  0.14  |
| **ViT DINOv2 ViTs14-reg**               |       0.61     |  0.46    |   0.69  |  0.50  |  0.26  |  0.81  |  0.46  |  0.24  |  0.11  |

## Example Images from Each Dataset with Labels
![alt text](https://github.com/broadinstitute/MorphEm/blob/main/example_image.png)
