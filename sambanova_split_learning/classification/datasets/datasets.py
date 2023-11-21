from torchvision.datasets import CIFAR10, ImageFolder


class CIFAR10Dataset(CIFAR10):
    def __getitem__(self, idx):
        return (*super().__getitem__(idx), {'id': idx})


class ImagenetDataset(ImageFolder):
    def __getitem__(self, idx):
        return (*super().__getitem__(idx), {'id': idx})
