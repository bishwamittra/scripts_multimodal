import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MammogramDataset(Dataset):
    def __init__(self, root_dir: str, dataset: str, transform=None, subclass: str = 'all', task_name: str = 'type'):
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.datapoints = []

        if subclass == 'all':
            prefix = ''
        elif subclass == 'calc':
            prefix = 'calc-'
        elif subclass == 'mass':
            prefix = 'mass-'
        with open(os.path.join(root_dir, prefix + dataset + '.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, each in enumerate(csv_reader):
                if i == 0:
                    continue
                if task_name == 'type':
                    self.datapoints.append((int(each[2] == "MALIGNANT"), os.path.join(root_dir, each[1])))
                elif task_name == 'form':
                    self.datapoints.append((int('Mass' in each[1]), os.path.join(root_dir, each[1])))
        random.shuffle(self.datapoints)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        import pydicom as dicom
        label, image_path = self.datapoints[idx]
        metadata = {"image_path": image_path}
        image = dicom.dcmread(image_path).pixel_array
        image = ((image / np.amax(image)) * 255).astype('uint8')
        x, y = image.shape
        new_x = int(1.6 * y)
        if new_x < x:
            image = image[int(x / 2 - new_x / 2):int(x / 2 + new_x / 2)]
        else:
            image = np.pad(image, [int((new_x - x) / 2), int((new_x - x) / 2)])
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), axis=2)

        if self.transform:
            image = self.transform(image)

        image = np.squeeze(image, 2)

        return image, label, metadata


class MammogramPatchDataset(Dataset):
    def __init__(self, root_dir, dataset, transform=None, subclass='all', task_name='type'):
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.datapoints = []
        if subclass == 'all':
            prefix = ''
        elif subclass == 'calc':
            prefix = 'calc_'
        elif subclass == 'mass':
            prefix = 'mass_'
        with open(os.path.join(root_dir, prefix + dataset + '_crop.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, each in enumerate(csv_reader):
                if i == 0:
                    continue
                if task_name == 'type':
                    self.datapoints.append((int(each[2] == "MALIGNANT"), os.path.join(root_dir, each[0])))
                elif task_name == 'form':
                    self.datapoints.append((int('Mass' in each[0]), os.path.join(root_dir, each[0])))
        random.shuffle(self.datapoints)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        import pydicom as dicom
        label, image_path = self.datapoints[idx]
        metadata = {"image_path": image_path}
        image = dicom.dcmread(image_path).pixel_array
        image = ((image / np.amax(image)) * 255).astype('uint8')
        x, y = image.shape
        if x > y:
            image = np.pad(image, [int((x - y) / 2), int((x - y) / 2)])
        elif y > x:
            image = np.pad(image, [int((y - x) / 2), int((y - x) / 2)])
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), axis=2)

        if self.transform:
            image = self.transform(image)

        image = np.squeeze(image, 2)

        return image, label, metadata
