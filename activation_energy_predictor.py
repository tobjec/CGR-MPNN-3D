import argparse
import os
from rdkit import Chem
import json
import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset
import torch_geometric as tg
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from ase.io import iread
from ase.visualize import view
from mace.calculators import mace_mp
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def process_xyz_to_npz(csv_file: Path, xyz_file: Path, npz_file: Path) -> None:
    """
    Process atomic coordinates from a .xyz file, align them with SMILES strings in a CSV file,
    and save the resulting molecular descriptors to a .npz file.

    Args:
        csv_file (Path): Path to the CSV file containing SMILES strings.
        xyz_file (Path): Path to the input .xyz file with atomic coordinates.
        npz_file (Path): Path to the output .npz file for saving molecular descriptors.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    device = "cpu" if torch.cuda.is_available() else "cpu"
    macemp = mace_mp(model="small", device=device)
    xyz_structures = []

    for atoms in tqdm(
        iread(xyz_file.as_posix()),
        desc="Extract MACE features",
        dynamic_ncols=True,
    ):
        xyz_structures.append(macemp.get_descriptors(atoms))

    smiles = pd.read_csv(csv_file.as_posix())
    features = []

    for i in range(len(smiles)):
        rsmi = smiles["smiles"][i].split(">")[0]
        r = Chem.MolFromSmiles(rsmi, params)
        ridx = np.array([a.GetAtomMapNum() - 1 for a in r.GetAtoms()])
        concat = [
            xyz_structures[3 * i][ridx, :],
            xyz_structures[3 * i + 1][ridx, :],
            xyz_structures[3 * i + 2][ridx, :],
        ]
        features.append(np.concatenate(concat, axis=1))

    np.savez(npz_file.as_posix(), *features)


def atom_features(atom: Chem.Atom) -> list:
    """
    Extracts features for an atom including atomic symbol, degree, charge,
    number of hydrogens, hybridization, aromaticity, and mass.

    Args:
        atom (rdkit.Chem.Atom): RDKit Atom object.

    Returns:
        list: Atom features as a one-hot encoding and continuous values.
    """
    features = (
        onek_encoding_unk(
            atom.GetSymbol(), ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
        )
        + onek_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
        + onek_encoding_unk(
            int(atom.GetHybridization()),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )
    return features


def bond_features(bond: Chem.Bond) -> list:
    """
    Extracts features for a bond including type, conjugation, and ring membership.

    Args:
        bond (rdkit.Chem.Bond): RDKit Bond object or None.

    Returns:
        list: Bond features as one-hot encoding and binary values.
    """
    bond_fdim = 7

    if bond is None:
        fbond = [1] + [0] * (bond_fdim - 1)  # Default feature for no bond
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # Bond exists
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
    return fbond


def onek_encoding_unk(value, choices: list) -> list:
    """
    Encodes a value as a one-hot vector with an additional entry for unknown values.

    Args:
        value: Value to encode.
        choices (list): List of possible values.

    Returns:
        list: One-hot encoded vector.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> dict:
    """
    Maps reactant atom indices to product atom indices based on atom map numbers.

    Args:
        mol_reac (rdkit.Chem.Mol): RDKit Molecule object for the reactant.
        mol_prod (rdkit.Chem.Mol): RDKit Molecule object for the product.

    Returns:
        dict: Mapping from reactant atom indices to product atom indices.
    """
    prod_map_to_id = dict(
        [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()]
    )
    reac_id_to_prod_id = dict(
        [
            (atom.GetIdx(), prod_map_to_id[atom.GetAtomMapNum()])
            for atom in mol_reac.GetAtoms()
        ]
    )
    return reac_id_to_prod_id


def make_mol(smi: str) -> Chem.Mol:
    """
    Converts a SMILES string to an RDKit Molecule with explicit hydrogens.

    Args:
        smi (str): SMILES representation of the molecule.

    Returns:
        rdkit.Chem.Mol: RDKit Molecule object.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smi, params)


class MolGraph:
    """
    Converts a molecule into a graph representation with atom and bond features.
    """

    def __init__(self, smiles: str):
        """
        Args:
            smiles (str): SMILES representation of the molecule.
        """
        self.smiles = smiles
        self.f_atoms = []  # List of atom features
        self.f_bonds = []  # List of bond features
        self.edge_index = []  # List of edge indices

        mol = make_mol(self.smiles)
        n_atoms = mol.GetNumAtoms()

        # Process atoms and bonds to generate features and edge indices
        for a1 in range(n_atoms):
            f_atom = atom_features(mol.GetAtomWithIdx(a1))
            self.f_atoms.append(f_atom)

            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)  # Add reverse bond
                self.edge_index.extend([(a1, a2), (a2, a1)])


class RxnGraph:
    """
    Converts a chemical reaction into a graph representation with atom and bond features.
    """

    def __init__(self, smiles: str):
        """
        Args:
            smiles (str): Reaction SMILES in the form "reactants>agents>products".
        """
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []  # List of atom features
        self.f_bonds = []  # List of bond features
        self.edge_index = []  # List of edge indices

        mol_reac = make_mol(self.smiles_reac)
        mol_prod = make_mol(self.smiles_prod)

        # Map reactant atom indices to product atom indices
        ri2pi = map_reac_to_prod(mol_reac, mol_prod)
        n_atoms = mol_reac.GetNumAtoms()

        # Process atoms and bonds for reactants and products
        for a1 in range(n_atoms):
            f_atom_reac = atom_features(mol_reac.GetAtomWithIdx(a1))
            f_atom_prod = atom_features(mol_prod.GetAtomWithIdx(ri2pi[a1]))
            f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
            f_atom = f_atom_reac + f_atom_diff
            self.f_atoms.append(f_atom)

            for a2 in range(a1 + 1, n_atoms):
                bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2])
                if bond_reac is None and bond_prod is None:
                    continue
                f_bond_reac = bond_features(bond_reac)
                f_bond_prod = bond_features(bond_prod)
                f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                f_bond = f_bond_reac + f_bond_diff
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)  # Add reverse bond
                self.edge_index.extend([(a1, a2), (a2, a1)])

class ChemDataset(Dataset):
    """
    Dataset class for chemical data, supporting molecule and reaction graph processing.

    Args:
        data_path (str): Path to the CSV file containing SMILES strings and labels.
        mode (str, optional): Processing mode ('mol' for molecules or 'rxn' for reactions). Defaults to 'rxn'.
    """

    def __init__(self, data_path: str, mode: str = "rxn", data_npz_path: str = None):
        """
        Initializes the dataset by reading the data, setting the mode, and preparing for graph generation.

        Args:
            data_path (str): Path to the CSV file containing SMILES and labels.
            mode (str): The mode for processing ('mol' or 'rxn'). Defaults to 'rxn'.
        """
        super().__init__()
        data_df = pd.read_csv(data_path)
        self.smiles = data_df.iloc[:, 0].values  # SMILES strings
        self.mode = mode  # Mode of processing: 'mol' or 'rxn'
        self.graph_dict = {}  # Cache for processed graph representations
        self.use_npz = False
        self.smi = None
        if data_npz_path:
            self.use_npz = True
            self.mace_features = {}
            with np.load(data_npz_path) as npz_data:
                for key in npz_data.files:
                    self.mace_features[key] = npz_data[key]

    def process_key(self, key: int) -> tg.data.Data:
        """
        Processes a single SMILES string at the specified index into a graph representation.

        Args:
            key (int): Index of the SMILES string to process.

        Returns:
            tg.data.Data: A PyTorch geometric Data object representing the molecule/reaction graph.
        """
        self.smi = self.smiles[key]
        if self.smi not in self.graph_dict:
            # Generate a graph depending on the processing mode
            if self.mode == "mol":
                molgraph = MolGraph(self.smi)
            elif self.mode == "rxn":
                molgraph = RxnGraph(self.smi)
            else:
                raise ValueError("Unknown option for mode", self.mode)
            # Convert the graph to PyTorch geometric Data
            mol = self.molgraph2data(molgraph, key)
            self.graph_dict[self.smi] = mol
        else:
            mol = self.graph_dict[self.smi]
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
        if self.use_npz:
            arr_key = key if key >= 0 else len(self.smiles) - key
            mace_feature = torch.from_numpy(self.mace_features[f"arr_{arr_key}"])
            data.x = torch.concatenate([data.x, mace_feature], dim=1)
        data.edge_index = (
            torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        )  # Edge connections
        data.edge_attr = torch.tensor(
            molgraph.f_bonds, dtype=torch.float
        )  # Bond features
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

def activation_energy_prediction(input_smiles: str, input_coordinates: str = "", output_results: str = "",
                                 print_results: bool = False, store_results: bool = False,
                                 output_format: str = "text") -> None:
    """
    Predict activation energies for chemical reactions based on SMILES and 3D coordinates.

    Args:
        input_smiles (str): Path to the file containing SMILES strings.
        input_coordinates (str): Path to the file containing 3D coordinates.
        output_results (str): Path to save the results.
        print_results (bool): Whether to print results to the console.
        store_results (bool): Whether to store results in a file.
        output_format (str): Format for stored results ("text" or "json"). Defaults to "text".
    """
    # Resolve input paths
    data_path_smiles = Path(input_smiles)
    data_path_coordinates = Path(input_coordinates)
    data_path_results = Path(output_results) if output_results else Path("results.txt")

    # Ensure proper output file name
    if data_path_results.is_dir():
        data_path_results /= "results.txt"

    # Validate input files
    if not data_path_smiles.is_file():
        raise FileNotFoundError(f"SMILES file not found: {data_path_smiles}")
    if not data_path_coordinates.is_file():
        raise FileNotFoundError(f"3D coordinates file not found: {data_path_coordinates}")

    # Convert XYZ to NPZ for processing
    data_path_npz = data_path_coordinates.parent / (data_path_coordinates.stem + '.npz')
    process_xyz_to_npz(data_path_smiles, data_path_coordinates, data_path_npz)

    # Load dataset and model
    pred_data = ChemDataset(data_path_smiles.as_posix(), data_npz_path=data_path_npz.as_posix())
    model_path = glob("cli_tool/**/*.pth")[0]
    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Prepare table and results list
    table = PrettyTable()
    table.field_names = ['SMILES', 'Activation Energy [kcal/mol]']
    results = []

    # Run predictions
    with torch.no_grad():
        for data in tqdm(pred_data, desc="Calculating Activation Energies", dynamic_ncols=True):
            activation_energy = model(data).item()
            results.append({'SMILES': data.smiles, 'Activation Energy': activation_energy})
            table.add_row([data.smiles, f"{activation_energy:.3f}"])

    # Output results
    if print_results:
        print("\nPredicted Activation Energies:\n")
        print(table)

    if store_results:
        if output_format == "text":
            with open(data_path_results, 'w') as f:
                f.write("Predicted Activation Energies:\n\n")
                f.write(str(table))
        elif output_format == "json":
            with open(data_path_results.with_suffix(".json"), 'w') as f:
                json.dump(results, f, indent=4)
        else:
            raise ValueError("Unsupported output format. Use 'text' or 'json'.")

        print(f"\nResults saved to: {data_path_results}")


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="CLI tool for predicting the activation energy of chemical reactions"+
                    " via the CGR MPNN 3D Graph Neural Network."
    )
    args.add_argument("--data_path_smiles", default="cli_tool/data/test.csv", type=str,
                      help="Path to .csv smiles data set.")
    args.add_argument("--data_path_coordinates", default="cli_tool/data/test.xyz", type=str,
                      help="Path to .xyz coordinates data set.")
    args.add_argument("--data_path_results", default="cli_tool/results.txt", type=str,
                      help="Path where results should be saved..")
    args.add_argument(
        "--data_path", default="datasets", type=str, help="Path to .csv data sets"
    )
    args.add_argument(
        "--store_results",
        action='store_true',
        help="Flag to save results",
    )
    args.add_argument(
        "--print_results",
        action='store_true',
        help="Flag to print results",
    )

    args = args.parse_args()

    activation_energy_prediction(args.data_path_smiles, args.data_path_coordinates, args.data_path_results,
                                 args.print_results, args.store_results)
