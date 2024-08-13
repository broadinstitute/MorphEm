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
from folded_dataset import fold_channels # not sure if this works
# reload(folded_dataset)


def configure_dataset(root_dir, dataset_name):
    df_path = f'{root_dir}/{dataset_name}/enriched_meta.csv'
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(csv_file=df_path, root_dir=root_dir, target_labels='train_test_split')

    return dataset


# Added ViTClass()




class ViTClass():
    def __init__(self, gpu):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

        dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        dinov2_vits14_reg.eval()
        dinov2_vits14_reg.to(self.device)

   
    def init_load_model(self):
        # device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        

        return self.dinov2_vits14_reg


    def refactor_img(base_path, root_dir):
        # not sure if this is right
        # base_path = "image path here" #change this to my path once I have images
        csv_path = os.path.join(base_path, "sc-metadata.csv")
        fold = folded_dataset.SingleCellDataset(csv_file=csv_path, root_dir=root_dir)
        # fold = fold_channels

        return fold



    def channel_to_rgb(channel):
        px = np.concatenate(
            (channel[np.newaxis, :, :], channel[np.newaxis, :, :], channel[np.newaxis, :, :]),
            axis=0)
        tensor = torch.Tensor(px)[None, ...]
        normalized_tensor = v2.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalized_tensor












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
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
      
        return x



class ResNetClass():
    def resnet_model(gpu):
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


        weights = ResNet18_Weights.IMAGENET1K_V1
        m = resnet18(weights=weights).to(device)
        feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]).to(device)


        return weights, feature_extractor



def get_save_features(feature_dir, root_dir, model_check, gpu, batch_size):
    dataset_names = ['Allen', 'CP', 'HPA']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    if model_check == "resnet":
        weights, feature_extractor = ResNetClass.resnet_model(gpu)
        preprocess = weights.transforms()
        feature_file = 'pretrained_resnet18_features.npy'

        
    else:
        
        convnext_instance = ConvNextClass(gpu)
        feature_file = 'pretrained_convnext_channel_replicate.npy'
        
        
    for dataset_name in dataset_names:
        dataset = configure_dataset(root_dir, dataset_name)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
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
        torch.cuda.empty_cache() # new line
        


def get_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="The root directory of the original images", required=True)
    parser.add_argument("--feat_dir", type=str, help="The directory that contains the features", required=True)
    parser.add_argument("--model", type=str, help="The type of model that is being trained and evaluated (convnext or resnet)", required=True, choices=['convnext', 'resnet'])
    parser.add_argument("--gpu", type=int, help="The gpu that is currently available/not in use", required=True)
    parser.add_argument("--batch_size", type=int, default=64, help="Select a batch size that works for your gpu size", required=True)
    
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

