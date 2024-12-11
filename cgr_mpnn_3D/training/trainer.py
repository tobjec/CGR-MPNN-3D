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
    """
    Trainer base class
    """

    @abstractmethod
    def train(self) -> None:
        """
        Overall training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> float:
        """
        Validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> float:
        """
        Training logic for one epoch.
        """

        pass


class RxnGraphTrainer(BaseTrainer):
    """
    Class to train reaction GNN model.
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
        """
        Constructor for the reaction GNN model trainer. 

        Args:
            name (str): Name of the file to be saved. 
            model (nn.Module): Initialised GNN model.
            optimizer (torch.optim): Optimizer for the parameter training.
            loss_fn (torch.nn): Used loss function
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            train_data (ChemDataset): Instance of training data class
            val_data (ChemDataset): Instance of validation data class
            device (torch.device): Processor to be used for training and evaluation.
            num_epochs (int): Number of epochs
            model_save_dir (str, optional): Path to folder where best model is saved.
                                            Defaults to "saved_models".
            batch_size (int, optional): Batch size. Defaults to 30.
            num_workers (int, optional): Number of parallel workers on CPU. Defaults to None.
            val_frequency (int, optional): Frequency of validation steps. Defaults to 5.
            logger (WandBLogger, optional): Logger to WandB server. Defaults to None.
        """
        
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
        self.train_loader = tg.loader.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        self.val_loader = tg.loader.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        
        # Standardizer
        self.stdizer_train = Standardizer(self.train_loader)
        self.stdizer_val = Standardizer(self.val_loader)

        # Wandb logger
        if self.logger:
            self.logger.watch(self.model)
                

    def _train_epoch(self, epoch_idx: int) -> float:
        """
        Training logic for one epoch

        Args:
            epoch_idx (int): Current training epoch.

        Returns:
            float: Root mean square error.
        """

        self.model.train()
        total_loss = 0.0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.loss_fn(predictions, self.stdizer_train(data.y))
            loss.backward()
            self.optimizer.step()
            total_loss += self.loss_fn(self.stdizer_train(predictions, rev=True), data.y).item()

        
        mean_loss = np.sqrt(total_loss / len(self.train_loader.dataset))

        if self.logger:
            self.logger.log({"train_loss": mean_loss, "epoch": epoch_idx})
        
        print(f"\n______epoch {epoch_idx}\nTrain loss, RMSE: {mean_loss:.4f}")
        return mean_loss


    def _val_epoch(self, epoch_idx: int) -> float:
        """
        Validation logic for one epoch.

        Args:
            epoch_idx (int): Current epoch.

        Returns:
            float: Root mean square error
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                predictions = self.model(data)
                loss = self.loss_fn(self.stdizer_val(predictions, rev=True), data.y)
                total_loss += loss.item()
                
        mean_loss = np.sqrt(total_loss / len(self.val_loader.dataset))

        if self.logger:
            self.logger.log({"val_loss": mean_loss, "epoch": epoch_idx})
        
        print(f"Val loss, RMSE: {mean_loss:.4f}\n")
        return mean_loss
        

    def train(self) -> dict:
        """
        Full training and validation logic of the trainer

        Returns:
            dict: Containing the RMSE for the training and validation steps.
        """
        
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