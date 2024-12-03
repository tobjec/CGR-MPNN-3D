import torch
from torch import nn
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
import tqdm 

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
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
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
                 train_data: ,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5) -> None:
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
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = Path(training_save_dir)
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.best_mean_per_class_accuracy = -1

        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Checks whether training dir exists
        self.training_save_dir.mkdir(parents=True, exist_ok=True)
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """

        self.model.train()
        total_loss = 0.0
        self.train_metric.reset()

        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.loss_fn(predictions, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.train_metric.update(predictions, labels)
        
        mean_loss = total_loss / len(self.train_loader)
        mean_accuracy, mean_per_class_accuracy = self.train_metric.accuracy(), self.train_metric.per_class_accuracy()
        
        print(f"______epoch {epoch_idx}\n\nTrain loss: {mean_loss:.4f}\n"+str(self.train_metric))
        return mean_loss, mean_accuracy, mean_per_class_accuracy


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        total_loss = 0.0
        self.val_metric.reset()
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)

                total_loss += loss.item()
                self.val_metric.update(predictions, labels)
                
        mean_loss = total_loss / len(self.val_loader)
        mean_accuracy, mean_per_class_accuracy = self.val_metric.accuracy(), self.val_metric.per_class_accuracy()
        
        print(f"______epoch {epoch_idx}\n\nVal loss: {mean_loss:.4f}\n"+str(self.val_metric))
        return mean_loss, mean_accuracy, mean_per_class_accuracy
        

    def train(self) -> dict:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """

        # Dictionary to track the losses and accuracies.
        train_dict = {'name': self.name,'train_losses': [], 'train_mean_a': [], 'train_mean_pca': [],
                      'val_losses': [], 'val_mean_a': [], 'val_mean_pca': []}

        for epoch_idx in tqdm.tqdm(range(self.num_epochs)):

            train_loss, train_acc, train_pc_acc = self._train_epoch(epoch_idx)
            train_dict['train_losses'].append((epoch_idx, train_loss))
            train_dict['train_mean_a'].append((epoch_idx, train_acc))
            train_dict['train_mean_pca'].append((epoch_idx, train_pc_acc))
            
            if epoch_idx % self.val_frequency == 0 or epoch_idx == self.num_epochs - 1:

                val_loss, val_acc, val_pc_acc = self._val_epoch(epoch_idx)
                train_dict['val_losses'].append((epoch_idx, val_loss))
                train_dict['val_mean_a'].append((epoch_idx, val_acc))
                train_dict['val_mean_pca'].append((epoch_idx, val_pc_acc))

                if val_pc_acc > self.best_mean_per_class_accuracy:
                    self.best_mean_per_class_accuracy = val_pc_acc
                    best_model_path = self.training_save_dir / f"{self.name}.pth"
                    self.model.save(best_model_path)
                    print(f"New best model with mpCA: {val_pc_acc:.4f} located at {best_model_path}")
            self.lr_scheduler.step()

        
        return train_dict