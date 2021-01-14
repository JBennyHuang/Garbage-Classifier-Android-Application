from glob import glob
from PIL import Image

import os
import shutil
import cv2
import random
import numpy as np
import torch as pt
import torchvision as tv

DATA_PATH = 'data/Garbage classification'

image_transforms = {
    'train':
    tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        tv.transforms.RandomRotation(degrees=15),
        tv.transforms.ColorJitter(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.CenterCrop(size=224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),

    'valid':
    tv.transforms.Compose([
        tv.transforms.Resize(size=256),
        tv.transforms.CenterCrop(size=224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test':
    tv.transforms.Compose([
        tv.transforms.Resize(size=256),
        tv.transforms.CenterCrop(size=224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class Dataset:
    def __init__(self, batch_size, ratio='6:2:2', data_path=DATA_PATH):

        split = [int(n) for n in ratio.split(':')]

        if not os.path.isdir(f'{data_path}/data'):
            groups = os.listdir(data_path)

            for group in groups:
                os.makedirs(f'{data_path}/data/train/{group}')
                os.makedirs(f'{data_path}/data/valid/{group}')
                os.makedirs(f'{data_path}/data/test/{group}')

            for group in groups:
                filenames = glob(f'{data_path}/{group}/*.jpg')
                length = len(filenames)
                filenames = iter(filenames)

                for _ in range(int(length * (split[0] / sum(split)))):
                    filepath = next(filenames)
                    shutil.copy(filepath, f'{data_path}/data/train/{group}/')
                
                for _ in range(int(length * (split[1] / sum(split)))):
                    filepath = next(filenames)
                    shutil.copy(filepath, f'{data_path}/data/valid/{group}/')

                for _ in range(int(length * (split[2] / sum(split)))):
                    filepath = next(filenames)
                    shutil.copy(filepath, f'{data_path}/data/test/{group}/')

        train_dataset = tv.datasets.ImageFolder(root=f'{data_path}/data/train', transform=image_transforms['train'])
        valid_dataset = tv.datasets.ImageFolder(root=f'{data_path}/data/valid', transform=image_transforms['valid'])
        test_dataset = tv.datasets.ImageFolder(root=f'{data_path}/data/test', transform=image_transforms['test'])

        self.train_dataloader = pt.utils.data.DataLoader(train_dataset, batch_size, True, num_workers=10)
        self.valid_dataloader = pt.utils.data.DataLoader(valid_dataset, batch_size, True, num_workers=10)
        self.test_dataloader = pt.utils.data.DataLoader(test_dataset, batch_size, True, num_workers=10)


    def next_train(self):
        for x_train, y_train in self.train_dataloader:
            yield x_train, y_train

    def next_valid(self):
        for x_valid, y_valid in self.valid_dataloader:
            yield x_valid, y_valid

    def next_test(self):
        for x_test, y_test in self.test_dataloader:
            yield x_test, y_test
