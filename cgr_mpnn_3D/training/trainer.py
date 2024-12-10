import torch
from torch import nn
import torch_geometric as tg
from abc import ABCMeta, abstractmethod
from pathlib import Path
import numpy as np
import os
from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.utils.standardizer import Standardizer
import tqdm
from wandb_logger import WandBLogger 

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> float:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> float:
        '''
        Holds training logic for one epoch.
        '''

        pass


class RxnGraphTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self,
                 name: str, 
                 model: nn.Module, 
                 optimizer: torch.optim,
                 loss_fn: torch.nn,
                 lr_scheduler: torch.optim.lr_scheduler,
                 train_data: ChemDataset,
                 val_data: ChemDataset,
                 device: torch.device,
                 num_epochs: int, 
                 model_save_dir: str = "saved_models",
                 batch_size: int = 30,
                 num_workers: int = None,
                 val_frequency: int = 5,
                 logger: WandBLogger=None) -> None:
        '''
        Args and Kwargs:
            name (str): Name of the model.
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        
        # Data members
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count() // 2 
        self.val_frequency = val_frequency
        self.best_val_loss = np.inf
        self.logger = logger

        # Data loaders
        self.train_loader = tg.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        self.val_loader = tg.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        
        # Standardizer
        self.stdizer = Standardizer(self.train_loader)

        # Wandb logger
        if self.logger:
            self.logger.watch(self.model)
                

    def _train_epoch(self, epoch_idx: int) -> float:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """

        self.model.train()
        total_loss = 0.0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.loss_fn(predictions, self.stdizer(data.y))
            loss.backward()
            self.optimizer.step()

            batch_size = data.y.size(0)
            total_loss += self.loss_fn(self.stdizer(predictions, rev=True), data.y).item() * batch_size
        
        mean_loss = np.sqrt(total_loss / len(self.train_loader))

        if self.logger:
            self.logger.log({"train_loss": mean_loss, "epoch": epoch_idx})
        
        print(f"\n______epoch {epoch_idx}\nTrain loss, RMSE: {mean_loss:.4f}")
        return mean_loss


    def _val_epoch(self, epoch_idx: int) -> float:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                predictions = self.model(data)
                loss = self.loss_fn(self.stdizer(predictions, rev=True), data.y)

                batch_size = data.y.size(0)

                total_loss += loss.item() * batch_size
                
        mean_loss = np.sqrt(total_loss / len(self.val_loader))

        if self.logger:
            self.logger.log({"val_loss": mean_loss, "epoch": epoch_idx})
        
        print(f"Val loss, RMSE: {mean_loss:.4f}\n")
        return mean_loss
        

    def train(self) -> dict:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """

        # Dictionary to track losses.
        train_dict = {'train_losses': [], 'val_losses': []}

        for epoch_idx in tqdm.tqdm(range(self.num_epochs)):

            train_loss = self._train_epoch(epoch_idx)
            train_dict['train_losses'].append(train_loss)
            
            if epoch_idx % self.val_frequency == 0 or epoch_idx == self.num_epochs - 1:

                val_loss = self._val_epoch(epoch_idx)
                train_dict['val_losses'].append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_path = self.model_save_dir / f"{self.name}.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"New best model with validation loss RMSE: {self.best_val_loss:.4f} located at {best_model_path}")
            self.lr_scheduler.step()
        
        if self.logger:
            self.logger.finish()

        return train_dict