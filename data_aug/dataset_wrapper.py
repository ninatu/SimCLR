import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from torch.utils.data import Dataset


from anomaly_detection.utils.data.datasets import Numpy2DDataset
from anomaly_detection.utils.data.transforms import Transform2D
np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, dataset_type, dataset_params, batch_size):
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.num_workers = dataset_params['num_workers']
        self.valid_size = dataset_params['valid_size']
        self.s = dataset_params['s']
        self.input_shape = eval(dataset_params['input_shape'])

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        transform = SimCLRDataTransform(data_augment)

        if self.dataset_type == 'STL10':
            train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
                                           transform=transform)
            print("Dataset: ", self.dataset_type, len(train_dataset), 'images')
            train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)

        elif self.dataset_type == 'MOOD':
            train_kwargs = self.dataset_params['train']
            val_kwargs = self.dataset_params['val']

            train_dataset = ConcatDataset([Numpy2DDataset(**params, transform=transform) for params in train_kwargs])
            val_dataset = ConcatDataset([Numpy2DDataset(**params, transform=transform) for params in val_kwargs])
            print("Dataset: ", self.dataset_type, len(train_dataset), 'train images', len(val_dataset), 'val images')

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, drop_last=True, shuffle=True)

            valid_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, drop_last=True)
        else:
            raise NotImplementedError()

        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def _get_simclr_pipeline_3d_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
