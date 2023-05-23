# MorphEm

MorphEm_70k v2.0 is ready to download at [here](https://caicedolab.s3.us-west-2.amazonaws.com/MorphEm/morphem_70k_2.0.zip).

## Requirements

We used Python 3.9.15 to run all the experiments. Set up the environment by running:
```
pip install -r <path-to-repo>/requirements.txt
conda install -c pytorch faiss-gpu
```


## Training and Testing
`morphem.ipynb` 
* Notebook example for loading and visualizing the data.

`Feature_Extraction.ipynb`
* File inputs:
    * `root_dir`: path to root directory
    * `dataset_name`: name of dataset, set to "HPA", "Allen", or "CP"
* Images are loaded with custom dataset loader in `folded_dataset.py`.
* Features are extracted with ResNet18 model pretrained on ImageNet images.
* By default, features are stored in `root_dir/features`, and contains one subfolder for each `dataset_name`.


`run_benchmark.py`
* Command line script to run the evaluation pipeline. Example command:
```
python run_benchmark.py --root_dir "../datasets/morphem_70k_2.0" 
                        --dest_dir "../results" \
                        --feature_dir "../datasets/morphem_70k_2.0/features" \
                        --feature_file "pretrained_resnet18_features.npy" \
                        --use_gpu \
                        --umap \
                        --classifier "knn"
```
 * `root_dir`: Path to root directory where the morphem dataset is stored.
 * `dest_dir`: Path to save results. Results include a csv files with mean accuracies for each task 
               and a dictionary for each dataset that stores more detailed results on specific classes.
 * `feature_dir`: Path to directory where features are stored. Within the feature folder there should be three subfolders, one for each dataset.
 * `feature_file`: Filename of features.
 * `use_gpu`: Use GPU for KNN computing.
 * `umap`: Create umap for features. 
 * `classifier`: Choice of classifier to use. Currently can be set to 'knn' or 'sgd'.
 


`Evaluation_pipeline.ipynb`
* Notebook version of `run_benchmark.py`
* File inputs:
    * `root_dir`: path to root directory
    * `dataset`: name of dataset, set to "HPA", "Allen", or "CP"
    * `leave_out`: name of leave-one-out task (e.g. "Task_four")
    * `leaveout_label`: column name of the leave out groups (e.g. "cell_type")
    * `model_choice`: type of classifier to use (i.e. "knn" or "sgd") 
* Notebook loads precomputed features (output of `Feature_Extraction.ipynb`) and train 
  a classifier to perform the validation task for each dataset. 
