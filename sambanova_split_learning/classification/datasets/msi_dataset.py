import collections
import os
from typing import List

import PIL
import torch
from skimage.io import imread
from torch.utils.data import Dataset

PIL.Image.MAX_IMAGE_PIXELS = None

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
DUMMY_LABEL = -1


class MSIDataset(Dataset):
    def __init__(self, data_dir: str, dataset: str, val_percent: int = 10, transform=None, dataseed: int = 0):

        assert dataset in ["train", "validation", "all", "test"], f"Invalid dataset:{dataset}"
        self.patient_ids = []
        self.transform = transform
        self.dataset = dataset
        pos_patient_to_image_path = {}
        neg_patient_to_image_path = {}
        for ddir in data_dir.split(","):
            if dataset == "test":
                pos_data_dir = os.path.join(ddir, "TEST_ALL_MSI")
                neg_data_dir = os.path.join(ddir, "TEST_ALL_MSS")
            else:
                pos_data_dir = os.path.join(ddir, "TRAIN_MSI")
                neg_data_dir = os.path.join(ddir, "TRAIN_MSS")

            pos_patient_to_image_path.update(self.get_patient_to_imagepath(pos_data_dir))
            neg_patient_to_image_path.update(self.get_patient_to_imagepath(neg_data_dir))

        self.patient_id_to_label = {patient_id: 1 for patient_id in pos_patient_to_image_path.keys()}
        self.patient_id_to_label.update({patient_id: 0 for patient_id in neg_patient_to_image_path.keys()})

        # Get patient ids based on dataset.
        if dataset in ["all", "test"]:
            self.patient_ids = list(pos_patient_to_image_path.keys()) + list(neg_patient_to_image_path.keys())
        else:
            pos_ids_train, pos_ids_val = self.split_train_val_patient_ids(patient_ids=list(
                pos_patient_to_image_path.keys()),
                                                                          val_percent=val_percent,
                                                                          seed=dataseed)

            neg_ids_train, neg_ids_val = self.split_train_val_patient_ids(patient_ids=list(
                neg_patient_to_image_path.keys()),
                                                                          val_percent=val_percent,
                                                                          seed=dataseed)
            if dataset == "train":
                self.patient_ids = pos_ids_train + neg_ids_train
            else:
                self.patient_ids = pos_ids_val + neg_ids_val

        patient_to_image_path = {**pos_patient_to_image_path, **neg_patient_to_image_path}

        # Create (patient_id, image_name, label) triplets
        self.metadata = []

        for patient_id in self.patient_ids:
            image_paths = patient_to_image_path[patient_id]
            label = self.patient_id_to_label[patient_id]
            self.metadata.extend([(patient_id, image_path, label) for image_path in image_paths])
        self.print_stats()

    def get_patient_to_imagepath(self, data_dir: str):
        mapping = collections.defaultdict(list)
        for image_path in os.listdir(data_dir):
            if image_path.endswith(".png"):
                if self.dataset == "test":
                    # We use the entire patient id during predict to get the coordinates for logging.
                    patient_id = "-".join(image_path.split("-"))
                else:
                    # During training/validation we need to chop off the coordinates to make sure we split correctly.
                    patient_id = "-".join(image_path.split("-")[:4])
                mapping[patient_id].append(os.path.join(data_dir, image_path))
        return mapping

    def split_train_val_patient_ids(self, patient_ids: List[str], val_percent: int, seed: int) -> List[str]:

        # Hard coding patient ids for validation to make sure we always use the same set across all image sizes and devices.
        gold_val_pids = [
            'TCGA-AG-A002-01Z', 'TCGA-AA-3715-01Z', 'TCGA-AG-3909-01Z', 'TCGA-G4-6320-01Z', 'TCGA-A6-6138-01Z',
            'TCGA-EI-6917-01Z', 'TCGA-A6-5665-01Z', 'TCGA-AZ-5403-01Z', 'TCGA-CL-4957-01Z', 'TCGA-AA-3971-01Z',
            'TCGA-AA-3867-01Z', 'TCGA-DY-A1DC-01Z', 'TCGA-AG-3885-01Z', 'TCGA-AG-A01W-01Z', 'TCGA-EI-6509-01Z',
            'TCGA-DC-6155-01Z', 'TCGA-CK-6751-01Z', 'TCGA-F4-6570-01Z', 'TCGA-DC-4749-01Z', 'TCGA-DM-A0X9-01Z',
            'TCGA-D5-6920-01Z', 'TCGA-D5-6539-01Z', 'TCGA-EI-6508-01Z', 'TCGA-AY-4071-01Z', 'TCGA-D5-6531-01Z',
            'TCGA-CM-5344-01Z'
        ]
        train_patients = []
        val_patients = []
        for patient_id in patient_ids:
            if patient_id in gold_val_pids:
                val_patients.append(patient_id)
            else:
                train_patients.append(patient_id)
        return train_patients, val_patients

    def print_stats(self):
        num_total_samples = len(self.metadata)
        num_positive_samples = len([label for _, _, label in self.metadata if label == 1])
        num_negative_samples = num_total_samples - num_positive_samples
        print(f"Stats for {self.dataset}")
        print(f"Total number of patient IDs: {len(self.patient_ids)}")
        print(f"Total number of samples: {num_total_samples}")
        print(f"positive: {num_positive_samples}, negative: {num_negative_samples}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> (torch.Tensor, int, str):
        if self.dataset in ["all"]:
            patient_id, image_path = self.metadata[idx]
            label = DUMMY_LABEL
        else:
            patient_id, image_path, label = self.metadata[idx]
        img = imread(image_path)
        if self.transform:
            img = self.transform(img)

        # Remove the 4th dimension
        img = img[:3, ...]

        metadata = {"patient_id": patient_id, "image_path": image_path}
        return img, int(label), metadata
