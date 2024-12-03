from rdkit import Chem

def atom_features(atom):
    features = onek_encoding_unk(atom.GetSymbol(), ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']) + \
        onek_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) + \
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4]) + \
        onek_encoding_unk(int(atom.GetHybridization()),[Chem.rdchem.HybridizationType.SP,
                                                        Chem.rdchem.HybridizationType.SP2,
                                                        Chem.rdchem.HybridizationType.SP3,
                                                        Chem.rdchem.HybridizationType.SP3D,
                                                        Chem.rdchem.HybridizationType.SP3D2
                                                        ]) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]
    return features

def bond_features(bond):
    bond_fdim = 7

    if bond is None:
        fbond = [1] + [0] * (bond_fdim - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return fbond

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def map_reac_to_prod(mol_reac, mol_prod):
    prod_map_to_id = dict([(atom.GetAtomMapNum(),atom.GetIdx()) for atom in mol_prod.GetAtoms()])
    reac_id_to_prod_id = dict([(atom.GetIdx(),prod_map_to_id[atom.GetAtomMapNum()]) for atom in mol_reac.GetAtoms()])
    return reac_id_to_prod_id

def make_mol(smi):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smi,params)

class MolGraph:
    def __init__(self, smiles):
        self.smiles = smiles
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        mol = make_mol(self.smiles)
        n_atoms=mol.GetNumAtoms()

        for a1 in range(n_atoms):
            f_atom = atom_features(mol.GetAtomWithIdx(a1))
            self.f_atoms.append(f_atom)

            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(a1, a2), (a2, a1)])

class RxnGraph:
    def __init__(self, smiles):
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        mol_reac = make_mol(self.smiles_reac)
        mol_prod = make_mol(self.smiles_prod)

        ri2pi = map_reac_to_prod(mol_reac, mol_prod)
        n_atoms = mol_reac.GetNumAtoms()

        for a1 in range(n_atoms):
            f_atom_reac = atom_features(mol_reac.GetAtomWithIdx(a1))
            f_atom_prod = atom_features(mol_prod.GetAtomWithIdx(a1))
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
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(a1, a2), (a2, a1)])

