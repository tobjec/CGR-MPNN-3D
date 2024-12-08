import torch
import numpy as np

class Standardizer:

    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.mean = np.mean(dataloader.dataset.labels)
        self.std = np.std(dataloader.dataset.labels)
    
    def __call__(self, x, rev=False):
        return (x-self.mean)/self.std if not rev else (x*self.std)+self.mean