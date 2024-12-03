import torch
import torch_geometric as tg
import pandas as pd
import numpy as np

from cgr_mpnn_3D.utils.graph_features import MolGraph, RxnGraph
 
from torch.utils.data import Dataset, DataLoader

class ChemDataset(Dataset):
    def __init__(self, smiles, labels, mode='mol'):
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
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

    def get(self,key):
        return self.process_key(key)

    def len(self):
        return len(self.smiles)

def construct_loader(data_path, shuffle=True, batch_size=50, mode='mol'):
    data_df = pd.read_csv(data_path)
    smiles = data_df.iloc[:, 0].values
    labels = data_df.iloc[:, 1].values.astype(np.float32)
    dataset = ChemDataset(smiles, labels, mode)
    loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            pin_memory=True,
                            sampler=None)
    return loader