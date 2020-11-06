import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data   = dataset[0].astype(np.float32)
        self.labels = dataset[1].astype(np.float32)
        self.size   = self.labels.shape[0]

    def __getitem__(self, idx):
        imgs = torch.from_numpy(np.asarray(self.data[idx,:,:,:]))
        lbls = torch.from_numpy(np.asarray(self.labels[idx,:,:,:]))
        return imgs, lbls

    def __len__(self):
        return len(self.labels)
