from torch.utils.data import Dataset
import torch
import random

class NoiseNotCifar10(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, perc_noise, cifar, label_noise=True, photo_noise=True):
        self.cifar_idxs = set(i for i in random.sample(range(0, len(cifar)), int((1-perc_noise)*len(cifar))))
        self.cifar = cifar
        self.label_noise = label_noise
        self.photo_noise = photo_noise

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        timg, tlabel = self.cifar[idx]
        if idx in self.cifar_idxs:
            return timg, tlabel
        generator = torch.Generator()
        generator.manual_seed(idx)
        gimg = torch.randn(3, 32, 32, generator=generator)
        glabel = torch.randint(0, 10, (1,), generator=generator).item()

        if self.label_noise and not self.photo_noise:
            return timg, glabel
        if not self.label_noise and self.photo_noise:
            return gimg, tlabel
        return gimg, glabel
    

class LengthCroppedAndShuffledDataset(Dataset):
    def __init__(self, dataset, length=None):
        super().__init__()
        self.dataset = dataset
        self.lenth = length
        if length is not None:
            self.idxs = [i for i in random.sample(range(0, len(dataset)), k=length)]
    
    def __len__(self):
        if self.lenth is not None:
            return self.lenth
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.lenth is not None:
            idx = self.idxs[idx] 
        return self.dataset[idx]