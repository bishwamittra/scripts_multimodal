import os

import torch.distributed as dist
from classification.datasets.classification_dataset import \
    ClassificationDataset
from classification.datasets.datasets import CIFAR10Dataset, ImagenetDataset
from classification.datasets.mammo_dataset import MammogramDataset
from classification.datasets.msi_dataset import MSIDataset
from classification.estimators.utils import compose_transforms, worker_init
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from samba_tools.constants import EstimatorMode


def get_dataset_csv_path(cfg, estimator_mode):
    try:
        if estimator_mode == EstimatorMode.PREDICT:
            return cfg.data.dataset_csv_path or os.path.join(cfg.data.data_dir, 'predict.csv')
        else:
            return cfg.data.dataset_csv_path or os.path.join(cfg.data.data_dir, 'labels.csv')
    except TypeError:
        if not cfg.run.listen_for_input:
            raise ValueError("data_dir and dataset_csv_path can only be None in online inference mode")


def get_dataset(cfg, subset: str = None) -> Dataset:
    if cfg.data is not None:
        transforms = cfg.data.transforms
    else:
        transforms = None

    if cfg.data.data_dir is not None:
        cfg.data.data_dir = str(cfg.data.data_dir)

    transform = compose_transforms(transform_config=transforms, mode=subset, use_bf16=(cfg.model.device == "RDU"))

    # Workaround setting a default for self.dataset_csv_path (davidku)
    dataset_csv_path = get_dataset_csv_path(cfg, estimator_mode=cfg.run.mode)

    # Get Dataset by task
    if cfg.task.task_name == "msi_v1":
        if subset == 'train':
            subset = 'validation' if cfg.regression_test.acc_test else 'train'
        dataset = MSIDataset(data_dir=cfg.data.data_dir, dataset=subset, transform=transform, dataseed=cfg.run.seed)

    elif cfg.task.task_name == "mammo":
        # Return the test set as training set for acc test to overfitting
        if subset == 'train' and not cfg.regression_test.acc_test:
            dataset = MammogramDataset(cfg.data.data_dir, "training", transform=transform)
        else:
            transform = compose_transforms(transform_config=transforms,
                                           mode='test',
                                           use_bf16=(cfg.model.device == "RDU"))
            dataset = MammogramDataset(cfg.data.data_dir, "testing", transform=transform)

    elif cfg.task.task_name == "cifar":
        dataset = CIFAR10Dataset(root=cfg.data.data_dir, train=(subset == "train"), download=True, transform=transform)

    elif cfg.task.task_name == "imagenet":
        if subset == 'train':
            subset = 'train' if not cfg.regression_test.acc_test else 'val'
        else:
            subset = 'val'
        dataset = ImagenetDataset(root=os.path.join(cfg.data.data_dir, subset), transform=transform)

    else:  # msi
        if subset == 'train':
            subset = 'train' if not cfg.regression_test.acc_test else 'validation'
        dataset = ClassificationDataset(dataset_csv_path,
                                        data_dir=cfg.data.data_dir,
                                        transform=transform,
                                        subset=subset)

    return dataset


def get_dataloader(cfg, dataset: Dataset, subset: str = None) -> DataLoader:
    """ Get dataloader from Dataset """
    if subset == "train":
        sampler = DistributedSampler(dataset, seed=cfg.run.seed) if dist.is_initialized() else RandomSampler(dataset)
    elif cfg.run.use_distributed_val and dist.is_initialized():
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    loader = DataLoader(dataset,
                        batch_size=cfg.model.batch_size,
                        drop_last=True,
                        sampler=sampler,
                        num_workers=cfg.run.num_workers,
                        worker_init_fn=worker_init,
                        persistent_workers=cfg.run.num_workers > 0,
                        prefetch_factor=cfg.data.prefetch_factor)

    return loader
