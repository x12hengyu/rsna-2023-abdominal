import os
from tqdm import tqdm

BASEDIR = '../rsna-2023-abdominal-trauma-detection'

TRAIN_IMG_PATH = os.path.join(BASEDIR, 'train_images')
TRAIN_META_PATH = os.path.join(BASEDIR, 'train_series_meta.csv')
TEST_IMG_PATH = os.path.join(BASEDIR, 'test_images')
TEST_META_PATH = os.path.join(BASEDIR, 'test_series_meta.csv')

TRAIN_LABEL_PATH = os.path.join(BASEDIR, 'train.csv')

img_paths = []
dcm_paths = []

print('Scanning directories...')
for patient in tqdm(os.listdir(TRAIN_IMG_PATH)):
    for scan in os.listdir(os.path.join(TRAIN_IMG_PATH, patient)):
        for img in os.listdir(os.path.join(TRAIN_IMG_PATH, patient, scan)):
            s = f"../rsna-2023-png/train_images/{patient}_{scan}_{img[:-4]}.png"
            img_paths.append(s)
            dcm_paths.append(os.path.join(TRAIN_IMG_PATH, patient, scan, img))

print(len(img_paths), len(dcm_paths))
img_paths1 = os.listdir('../rsna-2023-png/train_images')
print(len(img_paths1))
print(img_paths1)
print("/data/data5785/rsna-2023-png/train_images/59_22442_0000.png" in img_paths1)