import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from pathlib import Path
from cgr_mpnn_3D.training.trainer import RxnGraphTrainer
from cgr_mpnn_3D.models.GNN import GNN

# Adding working directory to PYTHONPATH (should be the project folder)
sys.path.append(os.getcwd())


class TestRxnGraphTrainer(unittest.TestCase):
    def setUp(self):
        # Mock dataset and data loaders
        mock_train_data = [
            Data(x=torch.randn(10, 5), y=torch.randn(10), labels=torch.randn(10)) for _ in range(10)
        ]
        mock_val_data = [
            Data(x=torch.randn(10, 5), y=torch.randn(10), labels=torch.randn(10)) for _ in range(5)
        ]

        self.train_loader = MagicMock()
        self.train_loader.__len__.return_value = len(mock_train_data)
        self.train_loader.dataset = mock_train_data

        self.val_loader = MagicMock()
        self.val_loader.__len__.return_value = len(mock_val_data)
        self.val_loader.dataset = mock_val_data

        # Mock model
        #self.model = MagicMock(spec=nn.Module)
        self.model = GNN(mock_train_data[0].num_node_features,
                         mock_train_data[0].num_edge_features)
        #self.model.return_value = torch.randn(10)

        # Optimizer, Loss, and Scheduler
        self.optimizer = Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.scheduler = MagicMock()

        # Trainer parameters
        self.device = torch.device("cpu")
        self.trainer = RxnGraphTrainer(
            name="test_model",
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            lr_scheduler=self.scheduler,
            train_data=mock_train_data,
            val_data=mock_val_data,
            device=self.device,
            num_epochs=2,
            batch_size=2,
            val_frequency=1,
            logger=None,
        )

        # Patch DataLoader to return mocked loaders
        self.trainer.train_loader = self.train_loader
        self.trainer.val_loader = self.val_loader

    def test_initialization(self):
        """Test that the trainer initializes correctly."""
        self.assertIsInstance(self.trainer, RxnGraphTrainer)
        self.assertEqual(self.trainer.batch_size, 2)
        self.assertEqual(self.trainer.device, self.device)


if __name__ == "__main__":
    unittest.main()
