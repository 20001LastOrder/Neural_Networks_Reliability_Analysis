
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import h5py


class ClevrDataset(Dataset):
    def __init__(self, root_dir, image_h5_filename, transform = None, questions=None):
        self.root_dir = root_dir
        self.filename = os.path.join(root_dir, image_h5_filename)
        with h5py.File(self.filename, 'r', swmr=True) as file:
            self.length = file["images"].shape[0]
        self.transform = transform
        self.image_h5 = None
        self.questions = questions

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        if self.image_h5 is None:
            image_h5 = h5py.File(self.filename, 'r',  swmr=True)
            self.images = image_h5['images']

        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image if self.questions is None else (image, self.questions[idx])

class ClevrDatasetWithLabels(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.filename = os.path.join(root_dir, 'train.h5' if train else 'test.h5')
        with h5py.File(self.filename, 'r', swmr=True) as file:
            self.length = file["images"].shape[0]
        self.transform = transform
        self.image_h5 = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.image_h5 is None:
            image_h5 = h5py.File(self.filename, 'r', swmr=True)
            self.images = image_h5['images']
            self.labels = image_h5['meta']

        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


class ClevrData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.224)

    def train_dataloader(self):
        transformation = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.hparams.image_height, self.hparams.image_width)),
                T.Normalize(self.mean, self.std)
            ]
        )

        dataset = ClevrDataset(self.hparams.data_dir, train=True,
                               transform=transformation)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        transformation = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.hparams.image_height, self.hparams.image_width)),
                T.Normalize(self.mean, self.std)
            ]
        )

        dataset = ClevrDataset(self.hparams.data_dir, train=False,
                               transform=transformation)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
