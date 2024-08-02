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


def configure_dataset(root_dir, dataset_name):
    df_path = f'{root_dir}/{dataset_name}/enriched_meta.csv'
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(csv_file=df_path, root_dir=root_dir, target_labels='train_test_split')

    return dataset





class ConvNextClass():
    def __init__(self, gpu):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
       
        pretrained = True
        model = create_model("convnext_tiny.fb_in22k", pretrained=pretrained).to(self.device)
        self.feature_extractor = nn.Sequential(
                            model.stem,
                            model.stages[0],
                            model.stages[1],
                            model.stages[2].downsample,
                            *[model.stages[2].blocks[i] for i in range(9)],
                            model.stages[3].downsample,
                            *[model.stages[3].blocks[i] for i in range(3)],
                            
                            )
        
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature_extractor, _, _ = ConvNextClass.convnext_model(gpu)
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
      
        return x

    # def convnext_model(gpu):
    #     instance = ConvNextClass(gpu)
    #     # feature_file = 'pretrained_convnext_channel_replicate.npy'

    #     return instance.feature_extractor

class ResNetClass():
    def resnet_model(gpu):
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


        # feature_file = 'pretrained_resnet18_features.npy'


        weights = ResNet18_Weights.IMAGENET1K_V1
        m = resnet18(weights=weights).to(device)
        feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]).to(device)

        # model_check = "resnet"

        return weights, feature_extractor



def get_save_features(feature_dir, root_dir, model_check, gpu, batch_size):
    dataset_names = ['Allen', 'CP', 'HPA']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    if model_check == "resnet":
        weights, feature_extractor = ResNetClass.resnet_model(gpu)
        preprocess = weights.transforms()
        feature_file = 'pretrained_resnet18_features.npy'
        # add feature file here and remove from class

        
    else:
        #feature_extractor, device, feature_file = ConvNextClass.convnext_model(gpu)
        #preprocess = None

        # add feature file here and remove from class
        
        convnext_instance = ConvNextClass(gpu) # reduce redudancy
        feature_file = 'pretrained_convnext_channel_replicate.npy'
        
        
    for dataset_name in dataset_names:
        dataset = configure_dataset(root_dir, dataset_name)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #reduce batch size to 128
        
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
                    feat_temp = convnext_instance.forward(expanded).cpu().detach().numpy()

                batch_feat.append(feat_temp)
                
            batch_feat = np.concatenate(batch_feat, axis=1)
            all_feat.append(batch_feat)
       
        all_feat = np.concatenate(all_feat)
        all_feat = all_feat.squeeze(2).squeeze(2)
        feature_path = feature_path = f'{feature_dir}/{dataset_name}/{feature_file}'
        np.save(feature_path, all_feat)


def get_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="The root directory of the original images", required=True)
    parser.add_argument("--feat_dir", type=str, help="The directory that contains the features", required=True)
    parser.add_argument("--model", type=str, help="The type of model that is being trained and evaluated (convnext or resnet)", required=True, choices=['convnext', 'resnet'])
    parser.add_argument("--gpu", type=int, help="The gpu that is currently available/not in use", required=True)
    parser.add_argument("--batch_size", type=int, default=192, help="Select a batch size that works for your gpu size", required=True)
    
    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    feat_dir = args.feat_dir
    model = args.model
    gpu = args.gpu
    batch_size = args.batch_size

    get_save_features(feat_dir, root_dir, model, gpu, batch_size)



























































# class ConvNextClass():
#     def convnext_model(gpu):
#         device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#         # self.feature_extractor = self.init_model()

#         # device can be removed
#         # init and init_model can be combined into one method

#     # def init_model(self):

#         pretrained = True
#         model = create_model("convnext_tiny.fb_in22k", pretrained=pretrained).to(device)
#         feature_extractor = nn.Sequential(
#                             model.stem,
#                             model.stages[0],
#                             model.stages[1],
#                             model.stages[2].downsample,
#                             *[model.stages[2].blocks[i] for i in range(9)],
#                             model.stages[3].downsample,
#                             *[model.stages[3].blocks[i] for i in range(3)],
#                         )

#         # model_check = "convnext"
#         return feature_extractor


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # feature_extractor, _, _ = ConvNextClass.convnext_model(gpu)
#         x = self.feature_extractor(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
      
#         return x

#     # def convnext_model():
#     #     instance = ConvNextClass()
#     #     # feature_file = 'pretrained_convnext_channel_replicate.npy'

#     #     return instance.feature_extractor

# class ResNetClass():
#     def resnet_model(gpu):
#         device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

#         # device is not needed in class since both models can use the same gpu (models are not used simultaneously)


#         # feature_file = 'pretrained_resnet18_features.npy'


#         weights = ResNet18_Weights.IMAGENET1K_V1
#         m = resnet18(weights=weights).to(device)
#         feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]).to(device)

#         # model_check = "resnet"

#         return weights, feature_extractord

