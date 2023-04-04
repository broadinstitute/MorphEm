# MorphEm

MorphEm_70k v2.0 is ready for download at {insert link}.

`morphem.ipynb` 
* Notebook example for loading and visualizing the data.

`Feature_Extraction.ipynb`
* File inputs:
    * `root_dir`: path to root directory
    * `dataset_name`: name of dataset, set to "HPA", "Allen", or "CP"
* Images are loaded with custom dataset loader in `folded_dataset.py`.
* Features are extracted with ResNet18 model pretrained on ImageNet images.


`Evaluation_pipeline.ipynb`
* File inputs:
    * `root_dir`: path to root directory
    * `dataset`: name of dataset, set to "HPA", "Allen", or "CP"
    * `leave_out`: name of leave-one-out task (e.g. "Task_four")
    * `leaveout_label`: column name of the leave out groups (e.g. "cell_type")
    * `model_choice`: type of classifier to use (i.e. "knn" or "sgd") 
* Notebook loads precomputed features (output of `Feature_Extraction.ipynb`) and train 
  a classifier to perform the validation task for each dataset. 

`folded_dataset.py` 
* Defines a custom dataset class to load the images and perform transformations.

`utils.py` 
* Defines the KNN classifier and other heavy lifting functions.
