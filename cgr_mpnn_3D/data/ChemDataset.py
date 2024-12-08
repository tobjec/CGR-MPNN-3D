import torch
import torch_geometric as tg
import pandas as pd
import numpy as np

from cgr_mpnn_3D.utils.graph_features import MolGraph, RxnGraph
 
from torch.utils.data import Dataset, DataLoader

class ChemDataset(Dataset):
    def __init__(self, data_path: str, mode='rxn'):
        super(ChemDataset, self).__init__()
        data_df = pd.read_csv(data_path)
        self.smiles = data_df.iloc[:,0].values
        self.labels = data_df.iloc[:,1].values.astype(np.float32)
        self.mode = mode

    def process_key(self, key):
        smi = self.smiles[key]
        if self.mode == 'mol':
            molgraph = MolGraph(smi)
        elif self.mode == 'rxn':
            molgraph = RxnGraph(smi)
        else:
            raise ValueError("Unknown option for mode", self.mode)
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        data.smiles = self.smiles[key]
        return data

    def __getitem__(self, key):
        return self.process_key(key)

    def __len__(self):
        return len(self.smiles)