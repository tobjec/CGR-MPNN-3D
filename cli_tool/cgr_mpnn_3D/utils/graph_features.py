from rdkit import Chem
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ase.io import iread
from mace.calculators import mace_mp


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
    device = "cpu"
    macemp = mace_mp(model="small", device=device, default_dtype="float32")
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
