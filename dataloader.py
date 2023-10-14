# %%
import os
import pickle
import random
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, RandomAffine, RandomErasing, ToTensor, Resize
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import numpy as np
import pandas as pd
import pydicom


BASEDIR = '../rsna-2023-abdominal-trauma-detection'

TRAIN_IMG_PATH = os.path.join(BASEDIR, 'train_images')
TRAIN_META_PATH = os.path.join(BASEDIR, 'train_series_meta.csv')
TEST_IMG_PATH = os.path.join(BASEDIR, 'test_images')
TEST_META_PATH = os.path.join(BASEDIR, 'test_series_meta.csv')

TRAIN_LABEL_PATH = os.path.join(BASEDIR, 'train.csv')


def fetch_img_paths():
    img_paths = []
    
    print('Scanning directories...')
    for patient in tqdm(os.listdir(TRAIN_IMG_PATH)):
        for scan in os.listdir(os.path.join(TRAIN_IMG_PATH, patient)):
            scans = []
            for img in os.listdir(os.path.join(TRAIN_IMG_PATH, patient, scan)):
                scans.append(os.path.join(TRAIN_IMG_PATH, patient, scan, img))
            img_paths.append(scans)
            
    return img_paths


def fetch_img_paths_png():
    img_paths = []
    
    all_pngs = sorted(os.listdir('../rsna-2023-png/train_images'))
    all_pngs = [os.path.join("../rsna-2023-png/train_images", d) for d in all_pngs]
    
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

def standardize_pixel_array(dicom_image):
    pixel_array = dicom_image.pixel_array
    
    if dicom_image.PixelRepresentation == 1:
        bit_shift = dicom_image.BitsAllocated - dicom_image.BitsStored
        dtype = pixel_array.dtype 
        new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dicom_image)

    if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = 1 - pixel_array
        
    # transform to hounsfield units
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope
    pixel_array = pixel_array * slope + intercept

    # windowing
    window_center = int(dicom_image.WindowCenter)
    window_width = int(dicom_image.WindowWidth)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    pixel_array = pixel_array.copy()
    pixel_array[pixel_array < img_min] = img_min
    pixel_array[pixel_array > img_max] = img_max

    # normalization
    if pixel_array.max() == pixel_array.min():
        pixel_array = np.zeros_like(pixel_array)  # Handle case of constant array
    else:
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())

    return pixel_array

def preprocess_png(png_path):
    img = cv2.imread(png_path)
    img = cv2.resize(img, (512, 512))
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    return greyscale

class AbdominalData(Dataset):
    
    def __init__(self, df_path, max_channel=4):
        
        super().__init__()
        
        # collect all the image instance paths
        self.img_paths = fetch_img_paths()
        self.max_channel = max_channel
        # self.max_channel = max([len(x) for x in self.img_paths])
        # print(len(self.img_paths), self.max_channel)
                
        df = pd.read_csv(df_path, index_col='patient_id')
        self.df_dict = df.to_dict(orient='index')
        for key, value in self.df_dict.items():
            self.df_dict[key] = list(value.values())
            
        df_meta = pd.read_csv(TRAIN_META_PATH)
        df_meta['ps'] = df_meta['patient_id'].astype(str) + "_" + df_meta['series_id'].astype(str)
        self.df_meta_dict = df_meta.set_index('ps')['incomplete_organ'].to_dict()
                
        self.transform = Compose([
                            RandomHorizontalFlip(),  # Randomly flip images left-right
                            ColorJitter(brightness=0.2),  # Randomly adjust brightness
                            ColorJitter(contrast=0.2),  # Randomly adjust contrast
                            RandomAffine(degrees=0, shear=10),  # Apply shear transformation
                            RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Apply zoom transformation
                            RandomErasing(p=0.2, scale=(0.02, 0.2)), # Coarse dropout
                        ])
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        dicom_images = self.img_paths[idx]
        center_idx = len(dicom_images) // 2
        dicom_images = dicom_images[center_idx - 2: center_idx + 2]

        patient_id = int(dicom_images[0].split('/')[-1].split('_')[0])
        series_id = int(dicom_images[0].split('/')[-1].split('_')[1])
        ps = str(patient_id) + "_" + str(series_id)
        # print(dicom_images[0], patient_id, series_id)
        
        images = []
        
        for d in dicom_images:
            image = preprocess_png(d)
            images.append(image)
        
        # padding
        if len(images) < self.max_channel:
            images.extend([np.zeros_like(images[0])] * (self.max_channel - len(images)))
        # while (len(images) != 4):
        #     images.append(torch.zeros_like(image))
        
        images = np.stack(images)
        image = torch.tensor(images, dtype = torch.float32).unsqueeze(dim = 1)
        image = self.transform(image).squeeze(dim = 1) # torch.Size([1727, 512, 512])
                
        label = self.df_dict[patient_id]
        incomplete_organ = self.df_meta_dict[ps]

        # labels
        bowel = np.argmax(label[0:2], keepdims = True)
        extravasation = np.argmax(label[2:4], keepdims = True)
        kidney = np.argmax(label[4:7], keepdims = False)
        liver = np.argmax(label[7:10], keepdims = False)
        spleen = np.argmax(label[10:], keepdims = False)
        
        # print(bowel, extravasation, kidney, liver, spleen, incomplete_organ)
        
        return image, {
            'bowel': bowel,
            'extravasation': extravasation,
            'kidney': kidney,
            'liver': liver,
            'spleen': spleen,
            'incomplete_organ': incomplete_organ
        }
    
class MetricsCalculator:
    
    def __init__(self, mode = 'binary'):
        
        self.probabilities = []
        self.predictions = []
        self.targets = []
        
        self.mode = mode
    
    def update(self, logits, target):
        """
        Update the metrics calculator with predicted values and corresponding targets.
        
        Args:
            predicted (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth targets.
        """
        if self.mode == 'binary':
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5)
        else:
            probabilities = F.softmax(logits, dim = 1)
            predicted = torch.argmax(probabilities, dim=1)
            
        self.probabilities.extend(probabilities.detach().cpu().numpy())
        self.predictions.extend(predicted.detach().cpu().numpy())
        self.targets.extend(target.detach().cpu().numpy())
    
    def reset(self):
        """Reset the stored predictions and targets."""
        
        self.probabilities = []
        self.predictions = []
        self.targets = []
    
    def compute_accuracy(self):
        """
        Compute the accuracy metric.
        
        Returns:
            float: Accuracy.
        """
        return accuracy_score(self.targets, self.predictions)
    
    def compute_auc(self):
        """
        Compute the AUC (Area Under the Curve) metric.
        
        Returns:
            float: AUC.
        """
        if self.mode == 'multi':
            return roc_auc_score(self.targets, self.probabilities, multi_class = 'ovo', labels=[0, 1, 2])
    
        else:
            return roc_auc_score(self.targets, self.probabilities)

# BATCH_SIZE = 256
# train_data_0, val_data_0 = AbdominalData(TRAIN_LABEL_PATH)
# 
# val_dataloader_0 = DataLoader(val_data_0,batch_size = BATCH_SIZE, shuffle = False)

# %%
data = AbdominalData(TRAIN_LABEL_PATH)
print(len(data))
print(sum([len(d) > 80 for d in data.img_paths]))

dataloader = DataLoader(data, batch_size=32, shuffle = True)