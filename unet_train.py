# %%
import os
import pickle
import random
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, RandomAffine, RandomErasing, ToTensor, Resize
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import numpy as np
import pandas as pd
import pydicom
import argparse

# %%
parser = argparse.ArgumentParser(description="Generate data on specified GPU devices.")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size of training.")
parser.add_argument("--epochs", type=int, required=True, help="Epochs of training.")
parser.add_argument("--lr", type=float, required=True, help="Learning rate of training.")
parser.add_argument("--device", type=int, required=True, help="List of GPU device numbers (e.g. 0).")
parser.add_argument("--subset", type=float, required=True, help="portion of data use")
args = parser.parse_args()
    
BATCH_SIZE = args.batch_size
device = torch.device(f'cuda:{args.device}')
torch.manual_seed(42)

# %%
BASEDIR = '/home/pranav/remote/xizheng/'

TRAIN_IMG_PATH = os.path.join(BASEDIR, 'train_images')
# TRAIN_META_PATH = os.path.join(BASEDIR, 'train_series_meta.csv')
# TEST_IMG_PATH = os.path.join(BASEDIR, 'test_images')
# TEST_META_PATH = os.path.join(BASEDIR, 'test_series_meta.csv')
TRAIN_LABEL_PATH = os.path.join('./', 'train.csv')

# %%
def fetch_img_paths_png():
    img_paths = []
    
    ppp = TRAIN_IMG_PATH
    # ppp = '/kaggle/input/rsna-abdominal-trauma-detection-png-pt1'
    
    all_pngs = sorted(os.listdir(ppp))
    all_pngs = [os.path.join(ppp, d) for d in all_pngs]
    
    cur_ps = []
    png = all_pngs[0]
    prev = png[:png.rfind('_')]
    
    for png in tqdm(all_pngs):
        patient_series = png[:png.rfind('_')]
        if prev == patient_series:
            cur_ps.append(png)
        else:
            img_paths.append(cur_ps)
            cur_ps = [png]
        prev = patient_series

    if cur_ps:  # to make sure the last group is added too
        img_paths.append(cur_ps)
    
    return img_paths

def preprocess_png(png_path):
    img = cv2.imread(png_path)
    img = cv2.resize(img, (512, 512))
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    return greyscale

# %% [markdown]
# ## Dataloader

# %%
def interpolate_channels(img_tensor):
    # Get the current number of channels
    C, H, W = img_tensor.shape

    # Initialize the output tensor
    output = torch.zeros((160, H, W))

    # Handle the edge case when C is 1
    if C == 1:
        for i in range(160):
            output[i] = img_tensor[0]
        return output

    # Handle the edge case when C is 2
    if C == 2:
        for i in range(80):
            output[i] = img_tensor[0]
        for i in range(80, 160):
            output[i] = img_tensor[1]
        return output

    # If channels are already 80 or more, return the original image
    if C >= 160:
        return img_tensor

    # Set the first and last channels
    output[0] = img_tensor[0]
    output[159] = img_tensor[-1]

    # Calculate the step for even spacing
    step = 158 / (C - 2)

    # Evenly space the remaining original channels in the range 1-78
    for i in range(1, C - 1):
        output[int(1 + i * step)] = img_tensor[i]

    # Perform linear interpolation
    for i in range(1, 159):
        if output[i].sum() == 0:
            left = i - 1
            right = i + 1
            while output[left].sum() == 0:
                left -= 1
            while output[right].sum() == 0:
                right += 1

            alpha = (i - left) / (right - left)

            output[i] = (1 - alpha) * output[left] + alpha * output[right]

    return output



class AbdominalData(Dataset):
    def __init__(self, df_path=TRAIN_LABEL_PATH, max_channel=4):
        super().__init__()
        
        # collect all the image instance paths
        self.img_paths = fetch_img_paths_png()
        self.max_channel = max_channel
                
        df = pd.read_csv(df_path, index_col='patient_id')
        
        ########## balance dataset ##########
        
        injured = df[df['any_injury'] == 1]
        not_injured = df[df['any_injury'] == 0]
        not_injured_sample = not_injured.sample(n=len(injured), random_state=1)  
        balanced_df = pd.concat([injured, not_injured_sample], axis=0)
        
        valid_patient_ids = set(balanced_df.index.astype(str))
        injured_ids = set(injured.index.astype(str))
        
        valid_img_paths = []

        num_injured, num_not_injured = 0, 0
        # Iterate through img_paths
        for paths in self.img_paths:
            # Example: from '/home/pranav/remote/xizheng/train_images/10004_21057_0000.png', extract '10004'
            patient_id = os.path.basename(paths[0]).split('_')[0]
            
            # Check if patient_id exists in valid_patient_ids
            if patient_id in valid_patient_ids:
                valid_img_paths.append(paths)
                
            if patient_id in injured_ids:
                num_injured += 1
            else:
                num_not_injured += 1
        
        self.img_paths = valid_img_paths
        
        ########## balance dataset ends ##########
                
        self.df_dict = balanced_df.to_dict(orient='index')
        for key, value in self.df_dict.items():
            self.df_dict[key] = list(value.values())

        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        dicom_images = self.img_paths[idx]
        
        patient_id = int(dicom_images[0].split('/')[-1].split('_')[0])
        series_id = int(dicom_images[0].split('/')[-1].split('_')[1])
        
        images = []
        
        for d in dicom_images:
            image = preprocess_png(d)
            images.append(image)
        
        images = np.stack(images)
        image = torch.tensor(images, dtype = torch.float32)
        image = interpolate_channels(image)
        center_idx = image.shape[0] // 2
        image = image[center_idx-80:center_idx+80:4]
                
        label = self.df_dict[patient_id]

        # labels
        bowel = label[0:2]
        extravasation = label[2:4]
        kidney = label[4:7]
        liver = label[7:10]
        spleen = label[10:13]
                
        return image, {
            'bowel': bowel,
            'extravasation': extravasation,
            'kidney': kidney,
            'liver': liver,
            'spleen': spleen,
        }

# %% [markdown]
# ## Net

# %%
data = AbdominalData()

subset_size = int(args.subset * len(data))
train_size = int(0.5 * subset_size)
val_size = subset_size - train_size
unused_size = len(data) - subset_size
train_data, val_data, _ = random_split(data, [train_size, val_size, unused_size])
# print(len(train_data), len(val_data), len(_))

train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
val_dataloader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)

# %%
from RSNA_model import RSNA_model

unet = RSNA_model().to(device)
optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

criterion_bowel = nn.BCEWithLogitsLoss().to(device)
criterion_extravasation = nn.BCEWithLogitsLoss().to(device)
criterion_kidney = nn.CrossEntropyLoss().to(device)
criterion_liver = nn.CrossEntropyLoss().to(device)
criterion_spleen = nn.CrossEntropyLoss().to(device)

# %%

from typing import Dict, List, Tuple
import pdb

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, scheduler: None,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X = X.to(device)
        N = X.shape[0]
        for k in y:
            y[k] = torch.stack(y[k]).transpose(0, 1).to(dtype=torch.float32)
            y[k] = y[k].to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss_b = criterion_bowel(y_pred[0], y["bowel"])
        loss_e = criterion_extravasation(y_pred[1], y["extravasation"])
        loss_k = criterion_kidney(y_pred[2], y["kidney"].argmax(dim=1))
        loss_l = criterion_liver(y_pred[3], y["liver"].argmax(dim=1))
        loss_s = criterion_spleen(y_pred[4], y["spleen"].argmax(dim=1))
        
        total_loss = loss_b + loss_e + loss_k + loss_l + loss_s
        train_loss += total_loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        total_loss.backward()

        # 5. Optimizer step
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Calculate and accumulate accuracy metric across all batches
        acc_b = (torch.argmax(y_pred[0], dim=1) == torch.argmax(y["bowel"], dim=1)).sum().item() / N
        acc_e = (torch.argmax(y_pred[1], dim=1) == torch.argmax(y["extravasation"], dim=1)).sum().item() / N
        acc_k = (torch.argmax(y_pred[2], dim=1) == torch.argmax(y["kidney"], dim=1)).sum().item() / N
        acc_l = (torch.argmax(y_pred[3], dim=1) == torch.argmax(y["liver"], dim=1)).sum().item() / N
        acc_s = (torch.argmax(y_pred[4], dim=1) == torch.argmax(y["spleen"], dim=1)).sum().item() / N
        
        train_acc += (acc_b + acc_e + acc_k + acc_l + acc_s) / 5

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X = X.to(device)
            N = X.shape[0]
            for k in y:
                y[k] = torch.stack(y[k]).transpose(0, 1).to(dtype=torch.float32) # [B, 2/3]        
                y[k] = y[k].to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss_b = criterion_bowel(test_pred_logits[0], y["bowel"])
            loss_e = criterion_extravasation(test_pred_logits[1], y["extravasation"])
            loss_k = criterion_kidney(test_pred_logits[2], y["kidney"].argmax(dim=1))
            loss_l = criterion_liver(test_pred_logits[3], y["liver"].argmax(dim=1))
            loss_s = criterion_spleen(test_pred_logits[4], y["spleen"].argmax(dim=1))
            
            total_loss = loss_b + loss_e + loss_k + loss_l + loss_s
            test_loss += total_loss.item()

            # Calculate and accumulate accuracy
            acc_b = (torch.argmax(test_pred_logits[0], dim=1) == torch.argmax(y["bowel"], dim=1)).sum().item() / N
            acc_e = (torch.argmax(test_pred_logits[1], dim=1) == torch.argmax(y["extravasation"], dim=1)).sum().item() / N
            acc_k = (torch.argmax(test_pred_logits[2], dim=1) == torch.argmax(y["kidney"], dim=1)).sum().item() / N
            acc_l = (torch.argmax(test_pred_logits[3], dim=1) == torch.argmax(y["liver"], dim=1)).sum().item() / N
            acc_s = (torch.argmax(test_pred_logits[4], dim=1) == torch.argmax(y["spleen"], dim=1)).sum().item() / N
            
            test_acc += (acc_b + acc_e + acc_k + acc_l + acc_s) / 5

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# Add writer parameter to train()
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: None,
          epochs: int,
          device: torch.device,
          ) -> Dict[str, List]:

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    print("Start Training: ")
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          optimizer=optimizer,
                                          scheduler = scheduler,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results


# %%
train(unet, train_dataloader, 
    val_dataloader, 
    optimizer,
    scheduler=scheduler,
    epochs=args.epochs,
    device=device,
)

torch.save(obj=unet.state_dict(), f=f"./unet_sub_{args.subset}_{args.batch_size}_{args.epochs}_{args.lr}.pth")


