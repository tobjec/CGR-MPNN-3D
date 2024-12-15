import unittest
from rdkit import Chem
import numpy as np
import sys
import os
from cgr_mpnn_3D.utils.graph_features import (
    atom_features,
    bond_features,
    onek_encoding_unk,
    MolGraph,
    RxnGraph,
    map_reac_to_prod,
    make_mol,
)

# Adding working directory to PYTHONPATH (should be the project folder)
sys.path.append(os.getcwd())


class TestGraphFeatures(unittest.TestCase):

    def test_bond_features(self):
        # Create a molecule with a single bond
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        features = bond_features(bond)
        self.assertEqual(len(features), 7)  # Total bond features
        self.assertEqual(features[1], 0)  # SINGLE bond
        self.assertEqual(features[2], 1)  # DOUBLE bond

    def test_onek_encoding_unk(self):
        # Test one-hot encoding
        choices = ["A", "B", "C"]
        encoding_known = onek_encoding_unk("A", choices)
        encoding_unknown = onek_encoding_unk("D", choices)
        self.assertEqual(encoding_known, [1, 0, 0, 0])
        self.assertEqual(encoding_unknown, [0, 0, 0, 1])

    def test_molgraph(self):
        # Test molecule graph construction
        smiles = "CCO"
        mol_graph = MolGraph(smiles)
        self.assertEqual(len(mol_graph.f_atoms), 3)  # 3 atoms in ethanol
        self.assertEqual(len(mol_graph.f_bonds), 4)  # 2 bonds * 2 (reverse bonds)
        self.assertEqual(len(mol_graph.edge_index), 4)  # 2 edges * 2 (reverse edges)

    def test_rxn_graph(self):
        # Test reaction graph construction
        smiles = "CCO>>CC=O"  # Ethanol to acetaldehyde
        rxn_graph = RxnGraph(smiles)
        self.assertEqual(len(rxn_graph.f_atoms), 3)  # 3 atoms in reactants
        self.assertEqual(len(rxn_graph.f_bonds), 4)  # 2 bonds * 2 (reverse bonds)
        self.assertEqual(len(rxn_graph.edge_index), 4)  # 2 edges * 2 (reverse edges)

        # Test atom mapping differences
        self.assertNotEqual(
            rxn_graph.f_atoms[0], rxn_graph.f_atoms[1]
        )  # Features differ

    def test_make_mol(self):
        # Test molecule creation with explicit hydrogens
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        mol_with_h = make_mol(smiles)
        self.assertEqual(
            mol_with_h.GetNumAtoms(), mol.GetNumAtoms()
        )  

    def test_map_reac_to_prod(self):
        # Test reactant to product atom mapping
        reac_smiles = "CCO"
        prod_smiles = "CC=O"
        mol_reac = make_mol(reac_smiles)
        mol_prod = make_mol(prod_smiles)
        mapping = map_reac_to_prod(mol_reac, mol_prod)
        self.assertEqual(mapping[0], 2)  # Atom indices should map correctly
        self.assertEqual(mapping[1], 2)


if __name__ == "__main__":
    unittest.main()
