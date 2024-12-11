import torch_geometric as tg
import numpy as np
from numpy.typing import ArrayLike

class Standardizer:
    """
    Class to calculate the mean and the standard deviation
    of a given data set.
    """

    def __init__(self, dataloader: tg.loader.DataLoader):
        """
        Args:
            dataloader (tg.loader.DataLoader): Dataloader to 
                                               extract data set.
        """
        self.mean = np.mean(dataloader.dataset.labels)
        self.std = np.std(dataloader.dataset.labels)
    
    def __call__(self, x: ArrayLike, rev=False) -> ArrayLike:
        """
        Args:
            x (ArrayLike): Original array
            rev (bool, optional): Inversion of the operation.
                                  Defaults to False.

        Returns:
            ArrayLike: Processed array 
        """
        return (x-self.mean)/self.std if not rev else (x*self.std)+self.mean