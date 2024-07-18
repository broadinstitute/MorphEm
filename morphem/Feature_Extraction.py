#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
import importlib
from tqdm import tqdm

from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model
import torch.nn.functional as F

import argparse



import folded_dataset
# reload(folded_dataset)

# In[ ]:


#pip install --upgrade pip


# In[ ]:


#pip install timm


# In[ ]:


# make sure to add your own directory for data directory (i.e. root directory) and feature directory




# In[ ]:


def configure_dataset(root_dir, dataset_name):
    df_path = f'{root_dir}/{dataset_name}/enriched_meta.csv'
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(csv_file=df_path, root_dir=root_dir, target_labels='train_test_split')

    return dataset


# In[ ]:

class ConvNextClass():
    def convnext_model():
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


        # In[ ]:


        feature_file = 'pretrained_convnext_channel_replicate.npy'
        # pooling = 'avg'


        # In[ ]:


        pretrained = True
        model = create_model("convnext_tiny.fb_in22k", pretrained=pretrained).to(device)
        feature_extractor = nn.Sequential(
                            model.stem,
                            model.stages[0],
                            model.stages[1],
                            model.stages[2].downsample,
                            *[model.stages[2].blocks[i] for i in range(9)],
                            model.stages[3].downsample,
                            *[model.stages[3].blocks[i] for i in range(3)],
                        )

        model_check = "convnext"
        # get_save_features(model, feature_dir, feature_file, root_dir, model_check)
        return feature_extractor, device


    def forward(x: torch.Tensor) -> torch.Tensor:
        # pooling = 'avg'
        feature_extractor, _ = ConvNextClass.convnext_model()
        x = feature_extractor(x)
        # if pooling == 'avg':
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # elif pooling == 'max':
        #     x = F.adaptive_max_pool2d(x, (1, 1))
        # elif pooling == 'avg_max':
        #     x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        #     x_max = F.adaptive_max_pool2d(x, (1, 1))
        #     x = torch.cat([x_avg, x_max], dim=1)
        # elif pooling == None:
        #     pass
        # else:
        #     raise ValueError(
        #         f"Pooling {self.cfg.pooling} not supported. Use one of {FeaturePooling.list()}"
        #     )
        # x = rearrange(x, "b c h w -> b (c h w)")
        return x


class ResNetClass():
    def resnet_model():
        device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")


        # In[ ]:


        feature_file = 'pretrained_resnet18_features.npy'


        # In[ ]:


        weights = ResNet18_Weights.IMAGENET1K_V1
        m = resnet18(weights=weights).to(device)
        feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]).to(device)

        model_check = "resnet"
        #get_save_features(m, feature_dir, feature_file, root_dir, model_check)

        return weights, feature_extractor, device



def get_save_features(feature_dir, feature_file, root_dir, model_check):
    dataset_names = ['CP','Allen','HPA']
    for dataset_name in dataset_names:
        dataset = configure_dataset(root_dir, dataset_name)
        train_dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        if model_check == "resnet":
            weights, feature_extractor, device = ResNetClass.resnet_model()
            preprocess = weights.transforms()
        else:
            _, device = ConvNextClass.convnext_model()
        all_feat = []
        for images, label in tqdm(train_dataloader, total=len(train_dataloader)):
            cloned_images = images.clone()
            batch_feat = []
            for i in range(cloned_images.shape[1]):
                # Copy each channel three times 
                channel = cloned_images[:, i, :, :]
                channel = channel.unsqueeze(1)
                expanded = channel.expand(-1, 3, -1, -1).to(device)
        
                if model_check == "resnet":
                    expanded = preprocess(expanded).to(device)
                    feat_temp = feature_extractor(expanded).cpu().detach().numpy()
                else: 
                    feat_temp = ConvNextClass.forward(expanded).cpu().detach().numpy()
                    
                batch_feat.append(feat_temp)
                
            batch_feat = np.concatenate(batch_feat, axis=1)
            all_feat.append(batch_feat)
       
        all_feat = np.concatenate(all_feat)
        all_feat = all_feat.squeeze(2).squeeze(2)
        feature_path = feature_path = f'{feature_dir}/{dataset_name}/{feature_file}'
        np.save(feature_path, all_feat)




# parser = argparse.ArgumentParser()
# parser.add_argument("root_dir", help="the root directory of the original images")
# parser.add_argument("feat_dir", help="the directory that contains the features")
# parser.add_argument("model", help="the type of model that is being trained and evaluated (convnext or resnet)")
# parser.add_argument("gpu", help="the gpu that is currently available/not in use", type=int)

# args = parser.parse_args()

if __name__ == '__main__':
    root_dir = '/scr/nnair/tiny_dataset/' # use only two images in a tiny directory with 2 images from each dataset (create directory called tiny_CHAMMI)
    feature_dir = '/scr/nnair/tiny_features' # use same structure as original directory (again a tiny directory for the features for the two images from each dataset)
    feature_file = 'pretrained_convnext_channel_replicate.npy'
    model_check = "convnext"

    get_save_features(feature_dir, feature_file, root_dir, model_check)

    # print("Features are all obtained and saved!")

    # Maybe feature_file might not need to be an argument since the feature_file is assigned in the respective classes
        # By doing this, this avoids the problem of mismatching models and feature file names (i.e. model is resnet, but feature_file is 
        # 'pretrained_convnext_channel_replicate.npy') 
            # this is just a suggestion, otherwise we could remove the feature_file variable being assigned in the classes and have it 
            # as an argument (for argparse)
