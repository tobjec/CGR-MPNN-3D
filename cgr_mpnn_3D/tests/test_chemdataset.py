import os
import tempfile
import unittest
import pandas as pd
import numpy as np
import torch
import sys
import torch_geometric as tg
from cgr_mpnn_3D.utils.graph_features import MolGraph, RxnGraph
from cgr_mpnn_3D.data.ChemDataset import ChemDataset

# Adding working directory to PYTHONPATH (should be the project folder)
sys.path.append(os.getcwd())


class TestChemDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and mock CSV file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_csv_path = os.path.join(self.temp_dir.name, "mock_data.csv")
        self.mock_npz_path = os.path.join(self.temp_dir.name, "mock_data.npz")

        # Create mock data
        self.mock_data = {
            "SMILES": [
                            "CCO>C>CCO",  # reactant>reaction center>product
                            "O>C>CO",
                            "N>C>CN",
                        ],
            "Label": [1.0, 2.0, 3.0],
        }
        self.mock_df = pd.DataFrame(self.mock_data)
        self.mock_df.to_csv(self.mock_csv_path, index=False)

        # Create mock .npz data
        mock_features = {
            f"arr_{i}": np.random.rand(3, 5)
            for i in range(len(self.mock_data["SMILES"]))
        }
        np.savez(self.mock_npz_path, **mock_features)

    def tearDown(self):
        # Cleanup temporary directory
        self.temp_dir.cleanup()

    def test_initialization(self):
        # Test dataset initialization
        dataset = ChemDataset(self.mock_csv_path, mode="rxn")
        self.assertEqual(len(dataset), len(self.mock_data["SMILES"]))
        self.assertEqual(dataset.mode, "rxn")
        self.assertEqual(dataset.smiles[0], self.mock_data["SMILES"][0])
        self.assertAlmostEqual(dataset.labels[0], self.mock_data["Label"][0])

    def test_process_key(self):
        # Test graph processing
        dataset = ChemDataset(self.mock_csv_path, mode="rxn")
        tg_data = dataset.process_key(0)
        self.assertIsInstance(tg_data, tg.data.Data)
        self.assertTrue(hasattr(tg_data, "x"))
        self.assertTrue(hasattr(tg_data, "edge_index"))
        self.assertTrue(hasattr(tg_data, "y"))
        self.assertTrue(hasattr(tg_data, "smiles"))

    def test_with_npz_features(self):
        # Test dataset with additional .npz features
        dataset = ChemDataset(
            self.mock_csv_path, mode="rxn", data_npz_path=self.mock_npz_path
        )
        tg_data = dataset.process_key(0)
        self.assertTrue(
            tg_data.x.shape[1] > 3
        )  # Check that MACE features are concatenated

    def test_getitem(self):
        # Test __getitem__ method
        dataset = ChemDataset(self.mock_csv_path, mode="rxn")
        tg_data = dataset[0]
        self.assertIsInstance(tg_data, tg.data.Data)
        self.assertEqual(tg_data.smiles, self.mock_data["SMILES"][0])

    def test_length(self):
        # Test __len__ method
        dataset = ChemDataset(self.mock_csv_path, mode="rxn")
        self.assertEqual(len(dataset), len(self.mock_data["SMILES"]))


if __name__ == "__main__":
    unittest.main()
