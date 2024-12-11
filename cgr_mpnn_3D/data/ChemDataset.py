import torch
import torch_geometric as tg
import pandas as pd
import numpy as np

from cgr_mpnn_3D.utils.graph_features import MolGraph, RxnGraph
from torch.utils.data import Dataset

class ChemDataset(Dataset):
    """
    Dataset class for chemical data, supporting molecule and reaction graph processing.

    Args:
        data_path (str): Path to the CSV file containing SMILES strings and labels.
        mode (str, optional): Processing mode ('mol' for molecules or 'rxn' for reactions). Defaults to 'rxn'.
    """

    def __init__(self, data_path: str, mode: str = 'rxn'):
        """
        Initializes the dataset by reading the data, setting the mode, and preparing for graph generation.

        Args:
            data_path (str): Path to the CSV file containing SMILES and labels.
            mode (str): The mode for processing ('mol' or 'rxn'). Defaults to 'rxn'.
        """
        super().__init__()
        data_df = pd.read_csv(data_path)
        self.smiles = data_df.iloc[:, 0].values  # SMILES strings
        self.labels = data_df.iloc[:, 1].values.astype(np.float32)  # Corresponding labels
        self.mode = mode  # Mode of processing: 'mol' or 'rxn'
        self.graph_dict = {}  # Cache for processed graph representations

    def process_key(self, key: int) -> tg.data.Data:
        """
        Processes a single SMILES string at the specified index into a graph representation.

        Args:
            key (int): Index of the SMILES string to process.

        Returns:
            tg.data.Data: A PyTorch geometric Data object representing the molecule/reaction graph.
        """
        smi = self.smiles[key]
        if smi not in self.graph_dict:
            # Generate a graph depending on the processing mode
            if self.mode == 'mol':
                molgraph = MolGraph(smi)
            elif self.mode == 'rxn':
                molgraph = RxnGraph(smi)
            else:
                raise ValueError("Unknown option for mode", self.mode)
            # Convert the graph to PyTorch geometric Data
            mol = self.molgraph2data(molgraph, key)
            self.graph_dict[smi] = mol
        else:
            mol = self.graph_dict[smi]
        return mol

    def molgraph2data(self, molgraph: RxnGraph | MolGraph, key: int) -> tg.data.Data:
        """
        Converts a molecule or reaction graph into a PyTorch geometric Data object.

        Args:
            molgraph (RxnGraph | MolGraph): The graph representation of the molecule or reaction.
            key (int): Index corresponding to the SMILES in the dataset.

        Returns:
            tg.data.Data: A PyTorch geometric Data object with atomic features, edges, and labels.
        """
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)  # Atomic features
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()  # Edge connections
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)  # Bond features
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)  # Target label
        data.smiles = self.smiles[key]  # SMILES string
        return data

    def __getitem__(self, key: int) -> tg.data.Data:
        """
        Retrieves a processed graph representation for a specific index.

        Args:
            key (int): Index of the data item.

        Returns:
            tg.data.Data: The PyTorch geometric Data object for the specified index.
        """
        return self.process_key(key)

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.smiles)