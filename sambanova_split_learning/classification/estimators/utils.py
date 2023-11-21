import pickle
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms as T
import yaml

import sambaflow.samba as samba

RESCALE_DIR = Path(__file__).parent.parent.absolute()


def load_data_pipeline(args):
    if Path(args.data_transform_config).is_absolute():
        data_transform_config = Path(args.data_transform_config)
    else:
        data_transform_config = RESCALE_DIR / args.data_transform_config

    with open(data_transform_config, "r") as f:
        transform_template = f.read()

    transform_template = transform_template.format(height=args.in_height, width=args.in_width)
    transforms = yaml.safe_load(transform_template)
    return transforms


def picklable_bf16_transform(img):
    return img.bfloat16()


def worker_init(worker_id):
    """
    Initialization routine to be executed at startup by each data loader.
    This should be passed to the data loader via the worker_init_fn parameter.
    """
    # limit the number of threads spawned by torch pthreadpool_create to avoid
    # contention in the data loader workers.
    # the default is the number of cores in the system
    torch.set_num_threads(1)


# TODO Remove this after we have all_gather in common utils
# Adapted from https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/distributed.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def all_gather(data, device='RDU'):
    """ All Gather for picklable objects
        Inputs:
            data: (object) a picklable object
            device: (str) 'RDU', 'CPU' or 'GPU'
        Returns: (list)
            data: gathered data using torch.distributed.all_gather
        """
    if not dist.is_initialized():
        raise Exception('Cannot perform all_gather when torch.distributed is not initialized')

    ws = dist.get_world_size()
    if ws == 1:
        return [data]

    device = 'cuda' if device == 'GPU' else 'cpu'
    if isinstance(data, (torch.Tensor, samba.SambaTensor)):
        tensor = data
        tensor_list = [torch.empty(tensor.shape, dtype=tensor.dtype, device=device) for _ in range(ws)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list

    # serialize data into a tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # find tensor size from each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(ws)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receive tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size, ), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size, ), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class TransformAdaptor:
    def __init__(self, transform, leagalizer=None):
        self.transform = transform
        self.leagalizer = leagalizer
        self.__doc__ = transform.__doc__

    def __call__(self, **kwargs):
        if self.leagalizer:
            kwargs = self.leagalizer(**kwargs)
        return self.transform(**kwargs)


def resize_leagalizer(**kwargs):
    from torchvision.transforms.functional import InterpolationMode
    interpolation_method = kwargs.get('interpolation', None)
    if interpolation_method:
        kwargs['interpolation'] = InterpolationMode[interpolation_method.upper()]
    return kwargs


def random_affine_leagalizer(**kwargs):
    kwargs = resize_leagalizer(**kwargs)
    return kwargs


def compose_transforms(transform_config, mode, use_bf16):

    try:
        config = transform_config[mode]
    except KeyError as e:
        msg = f"No pipeline specified for mode {mode} in config {transform_config.keys()}"
        raise KeyError(msg) from e

    transform_list = []
    for key in config:
        if isinstance(config[key], bool):
            xform = TRANSFORM_REGISTRY[key]()
        elif isinstance(config[key], dict):
            xform = TRANSFORM_REGISTRY[key](**config[key])
        else:
            raise ValueError(f"Unsupported config type given for {key}: {type(config[key])}")
        transform_list.append(xform)

    if use_bf16:
        bf16_conversion = T.Lambda(picklable_bf16_transform)
        transform_list.append(bf16_conversion)

    return T.Compose(transform_list)


TRANSFORM_REGISTRY = {
    'ToPILImage': T.ToPILImage,
    'Resize': TransformAdaptor(T.Resize, resize_leagalizer),
    'RandomHorizontalFlip': T.RandomHorizontalFlip,
    'RandomVerticalFlip': T.RandomVerticalFlip,
    'RandomRotation': T.RandomRotation,
    'Normalize': T.Normalize,
    'ToTensor': T.ToTensor,
    'RandomResizedCrop': T.RandomResizedCrop,
    'CenterCrop': T.CenterCrop,
    'RandomAffine': TransformAdaptor(T.RandomAffine, random_affine_leagalizer),
    'RandomCrop': T.RandomCrop,
}
