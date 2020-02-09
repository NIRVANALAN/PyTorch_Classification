import numpy as np
import pdb
import random
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import torchvision
from PIL import Image
from tqdm import tqdm, trange

import pathlib
from pathlib import Path

from sklearn.model_selection import train_test_split

category = ['NORMAL', 'RCM', 'DCM', 'HCM']

num_classes = 4


def mkdir_if_not(file_dir):
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def load_pil(img, shape=None):
    img = Image.open(img)
    if shape:
        img = img.resize((shape, shape), Image.BILINEAR)
    return np.array(img)


def retrieve_imgs(subclass_root_dir, split_list, dataset_list, label):
    subclass_datalist = []
    for slide_collection in split_list:
        current_path = subclass_root_dir.joinpath(slide_collection)
        for slide in os.listdir(current_path):
            if int(slide.split('-')[1]) < 3:
                slide_path = current_path.joinpath(slide)
                for sub_area in os.listdir(slide_path):
                    if sub_area is not 'Heart_trabe':
                        sub_area_path = slide_path.joinpath(sub_area)
                        imgs = os.listdir(sub_area_path)
                        imgs = [
                            f'{str(sub_area_path.joinpath(img))} {label}\n' for img in imgs]
                        subclass_datalist.extend(imgs)

    dataset_list.append(subclass_datalist)


def flatten(l): return [
    item for sublist in l for item in sublist[:]]  # * interesting


def shuffle(l): return random.shuffle(l)


def generate_list(root_dir, test_size=0.3, unit=1.5e4):
    train_list = []
    test_list = []
    for subclass in range(len(category)):
        current_path = Path(root_dir, category[subclass])
        patients = sorted(os.listdir(current_path))
        test_list_len = int(len(patients)*test_size)
        test_split, train_split = patients[-test_list_len:], patients[:-test_list_len]
        retrieve_imgs(current_path, train_split, train_list, subclass)
        retrieve_imgs(current_path, test_split, test_list, subclass)

    for sublist in train_list + test_list:
        shuffle(sublist)

    train_list, test_list = flatten(train_list), flatten(test_list)
    # with open('train_list.txt', 'w') as train_img_list:
    #     train_img_list.writelines(train_list)
    with open('new_test_list.txt', 'w') as train_img_list:
        train_img_list.writelines(test_list)


#
root_dir = '/home/zhourongchen/lys/RCM_DATASET/'
generate_list(root_dir)
