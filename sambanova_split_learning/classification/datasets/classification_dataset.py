import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from skimage.io import imread
from torch.utils.data import Dataset

DUMMY_LABEL = -1


class ClassificationDataset(Dataset):
    """
    Generic classification dataset.

    Builds a dataset from the CSV at `data_csv_path`, which should be formatted like:
        image_path, label, subset, metadata
    including a header. Label should be an integer.
    If data_dir is supplied, image_path is assumed to be relative to that directory.

    Args:
        data_csv_path (str): Path to the dataset CSV
        data_dir (str): Path to data directory (for relative image paths)
        subset (str): One of ["all", "train", "validation", "predict"]
        transform (Any): Transforms to apply to image
        image_read_fn (Any): Function to read image from path
    """

    required_csv_columns = ["image_path", "label", "subset", "metadata"]

    def __init__(self,
                 data_csv_path: str,
                 data_dir: str = None,
                 subset: str = "train",
                 transform: Any = None,
                 image_read_fn: Any = None):
        subsets = ["all", "train", "validation", "test"]
        if subset not in subsets:
            raise ValueError(f'subset must be one of: {subsets}. Got "{subset}".')
        if data_dir is not None and not os.path.exists(data_dir):
            raise RuntimeError(f'Data directory does not exist: "{data_dir}".')
        if not os.path.exists(data_csv_path):
            raise RuntimeError(f'Dataset csv path does not exist: "{data_csv_path}".')

        self.transform = transform
        self.subset = subset
        self.data_dir = data_dir
        self.data_csv = pd.read_csv(data_csv_path)
        self.image_read_fn = image_read_fn or imread
        for col in ClassificationDataset.required_csv_columns:
            if subset == "all" and col in ["label", "subset"]:
                continue
            assert col in self.data_csv.columns, f"Missing required column {col} from dataset CSV"
        if subset != "all":
            self.data_csv = self.data_csv.loc[self.data_csv["subset"] == subset].reset_index()
        # We want the labels to be ints, but just check that they're numeric to guard against strings, etc.
        if "label" in self.data_csv.columns:
            assert is_numeric_dtype(self.data_csv["label"]), \
            f"Label column loaded as type {self.data_csv['label'.dtype]}, expected int"

    def __len__(self) -> int:
        return len(self.data_csv)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label, metadata)
        """
        sample = self.data_csv.iloc[idx]
        image_path = sample["image_path"]
        patient_id = sample["metadata"]
        if self.data_dir is not None:
            image_path = os.path.join(self.data_dir, image_path)
        image = self.image_read_fn(image_path)

        # Some images only return (H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        # In case num_channels=1
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))

        # Run transformation functions
        image = self.transform(image)
        metadata = {"image_path": image_path, "patient_id": patient_id}
        if self.subset == "all":
            return image, DUMMY_LABEL, metadata
        return image, sample["label"], metadata
