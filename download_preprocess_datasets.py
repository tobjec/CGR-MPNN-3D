import urllib.request
import numpy as np
import torch
from transition1x import Dataloader
from tqdm import tqdm
import os
import ase
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import pandas as pd
import shutil
from pathlib import Path
from rdkit import Chem
from mace.calculators import mace_mp

def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """
    Display the progress of a file download as a percentage.

    Args:
        block_num (int): The current block number being downloaded.
        block_size (int): The size of each block in bytes.
        total_size (int): The total size of the file in bytes.
    """
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    print(f"\rDownloading: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="")


def process_rxn_files(base_folder: Path, rxn_range: int) -> tuple:
    """
    Process .log files in the specified reaction range to generate corresponding .xyz files
    and return lists of ASE atoms for reactants and products.

    Args:
        base_folder (Path): Path to the folder containing reaction directories.
        rxn_range (int): Range of reaction indices to process.

    Returns:
        tuple: Two lists containing ASE Atoms objects for reactants and products, respectively.
    """
    rrs = []
    pps = []

    for i in range(rxn_range):
        rxn_id = f'{i:06d}'
        rxn_folder = base_folder / f'rxn{rxn_id}'
        
        # Process reactant
        process_log_to_xyz(rxn_folder / f'r{rxn_id}.log', rxn_folder / f'r{rxn_id}.xyz')
        
        # Process product
        process_log_to_xyz(rxn_folder / f'p{rxn_id}.log', rxn_folder / f'p{rxn_id}.xyz')

        rrs.append(ase.io.read(rxn_folder / f'r{rxn_id}.xyz'))
        pps.append(ase.io.read(rxn_folder / f'p{rxn_id}.xyz'))
    
    return rrs, pps

def process_log_to_xyz(log_file: Path, xyz_file: Path) -> None:
    """
    Extract atomic coordinates and other relevant data from a .log file
    and write it to a .xyz file.

    Args:
        log_file (Path): Path to the input .log file.
        xyz_file (Path): Path to the output .xyz file.
    """
    log_file = Path(log_file)
    xyz_file = Path(xyz_file)
    
    try:
        # Read the log file
        with log_file.open('r') as f:
            lines = f.readlines()
        
        # Extract the number of atoms (NAtoms)
        natom = None
        for i,line in enumerate(lines):
            if 'NAtoms' in line:
                natom = int(lines[i+1].split()[0])
                break
        
        molecule_start = None
        for i,line in enumerate(lines):
            if '$molecule' in line:
                molecule_start = i + 2
                break
        molecule_data = lines[molecule_start:molecule_start + natom]

        # Write to the .xyz file
        with xyz_file.open('w') as f:
            f.write(f"{natom}\n\n")  
            f.writelines(molecule_data)
    
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #macemp = MACECalculator(model_paths=[mace_model_params_path],
    #                        device=device, default_dtype="float32")
    macemp = mace_mp(model="small", device=device)
    xyz_structures = []

    for atoms in tqdm(ase.io.iread(xyz_file.as_posix()), desc="Extract MACE features", dynamic_ncols=True):
        xyz_structures.append(macemp.get_descriptors(atoms))

    smiles = pd.read_csv(csv_file.as_posix())
    features = []

    for i in range(len(smiles)):
        rsmi = smiles['smiles'][i].split('>')[0]
        r = Chem.MolFromSmiles(rsmi, params)
        ridx = np.array([a.GetAtomMapNum()-1 for a in r.GetAtoms()])
        concat = [xyz_structures[3*i][ridx,:], xyz_structures[3*i+1][ridx,:], xyz_structures[3*i+2][ridx,:]]
        features.append(np.concatenate(concat, axis=1))

    np.savez(npz_file.as_posix(), *features)


class PreProcessTransition1x:
    """
    Preprocesses the Transition1x dataset, including downloading, extracting,
    and creating structured data files.
    """

    def __init__(self, fpath_download: str='downloaded_datasets', fpath_processed: str='datasets_test',
                 dlink_transition: str='https://figshare.com/ndownloader/files/36035789/transition1x.h5',
                 dlink_wb97xd3_csv: str='https://zenodo.org/records/3715478/files/wb97xd3.csv',
                 dlink_wb97xd3: str='https://zenodo.org/records/3715478/files/wb97xd3.tar.gz',
                 rxn_range: int=11961, keep_downloads: bool=False):
        """
        Args:
            fpath_download (str, optional): Directory to save downloaded files.
                                            Defaults to 'downloaded_datasets'.
            fpath_processed (str, optional): Directory to save processed datasets.
                                            Defaults to 'datasets_test'.
            dlink_transition (str, optional): URL for Transition1x data file.
                                            Defaults to official source.
            dlink_wb97xd3_csv (str, optional): URL for WB97XD3 CSV file.
                                            Defaults to official source.
            dlink_wb97xd3 (str, optional): URL for WB97XD3 tarball file.
                                        Defaults to official source.
            rxn_range (int, optional): Number of reactions to process. Defaults to 11961.
            keep_downloads (bool, optional): Whether to retain downloaded files. Defaults to False
        """
        
        self.fpath_download = Path(fpath_download)
        self.fpath_processed = Path(fpath_processed)
        self.dlink_transition = dlink_transition
        self.dlink_wb97xd3 = dlink_wb97xd3
        self.dlink_wb97xd3_csv = dlink_wb97xd3_csv
        self.fpath_wb97xd3_csv = self.fpath_download / os.path.basename(self.dlink_wb97xd3_csv)
        self.fpath_transition = self.fpath_download / os.path.basename(self.dlink_transition)
        self.folder_to_extract = self.fpath_download / os.path.basename(self.dlink_wb97xd3).split('.')[0]
        self.rxn_range = rxn_range
        self.keep_downloads = keep_downloads

    def start_data_acquisition(self, data_sets: list=['train', 'val', 'test']):
        """
        Orchestrates the entire data acquisition process, including folder creation,
        file downloads, extraction, and dataset generation.

        Args:
            data_sets (list, optional): Dataset splits to generate. Defaults to ['train', 'val', 'test'].
        """
        self.create_folders()
        self.download_files()
        print(f'Start of the extraction of {self.folder_to_extract}')
        self.extract_files()
        print('End of the extraction')
        print('Beginning of the dataset creation.')
        self.create_dataset_files(data_sets)

    def download_files(self) -> None:
        """
        Downloads required dataset files from predefined URLs if not already present.
        """
        
        for url in [self.dlink_transition, self.dlink_wb97xd3, self.dlink_wb97xd3_csv]:
            
            filename = os.path.basename(url)
            file_path = self.fpath_download / filename

            if not file_path.exists():
                file_path.touch()
                print(f'Downloading: {filename}')
                urllib.request.urlretrieve(url, file_path.as_posix(), reporthook=progress_callback)
                print(f'Finished downloading: {filename}', end='\n\n')
        
    def extract_files(self) -> None:
        """
        Extracts compressed files downloaded for processing.
        """

        filenames = [os.path.basename(file) for file in [os.path.basename(self.dlink_wb97xd3)]]
        for filename in filenames:
            fpath = self.fpath_download / filename
            folder_to_extract = self.fpath_download
            shutil.unpack_archive(fpath, folder_to_extract)
            print(f'File {filename} extracted to {folder_to_extract.as_posix()}')
    
    def create_dataset_files(self, data_sets: list) -> None:
        """
        Processes reaction files, generates .xyz files, and creates datasets for specified splits.

        Args:
            data_sets (list): Dataset splits to process, e.g., ['train', 'val', 'test'].
        """
        
        rrs, pps = process_rxn_files(self.folder_to_extract, self.rxn_range)
        d = pd.read_csv(self.fpath_wb97xd3_csv)

        for split in tqdm(data_sets, desc='Data Splits', dynamic_ncols=True):

            dataloader = Dataloader(self.fpath_transition, only_final=True, datasplit=split)

            all_structures = []
            e_a = []
            e_a_original = []
            rxns = []
            for molecule in tqdm(dataloader, desc=f'Data records {split}', dynamic_ncols=True):
                #Make XYZ, energies, forces:
                for s in ["reactant", "transition_state", "product"]:
                    atoms = Atoms(molecule[s]["atomic_numbers"])
                    atoms.set_positions(molecule[s]["positions"])
                    results = {"energy": molecule[s]["wB97x_6-31G(d).energy"], "forces": molecule[s]["wB97x_6-31G(d).forces"]}
                    atoms.calc = SinglePointCalculator(atoms, **results)
                    all_structures.append(atoms)

                #Make reaction SMILES, activation energies
                ts_energy = molecule["transition_state"]["wB97x_6-31G(d).energy"]
                r_energy = molecule["reactant"]["wB97x_6-31G(d).energy"]
                # Transformed from eV to kcal/mol
                e_a.append((ts_energy - r_energy)*23.06)

                #Unfortunately, the files are mismatched, and T1x labels consecutively from 0 to 10072 instead of the actual rxn indices
                candidates = []
                r = all_structures[-3]
                p = all_structures[-1]
                for i,rr in enumerate(rrs):
                    if "".join([str(n) for n in r.symbols.numbers]) == "".join([str(n) for n in rr.symbols.numbers]):
                        candidates.append(i)
                id = candidates[np.argmin([np.linalg.norm(p.positions - pps[i].positions) for i in candidates])]
                rxns.append(d[d['idx']==id]['rsmi'].values[0] + ">>" + d[d['idx']==id]['psmi'].values[0])
                e_a_original.append(d[d['idx']==id]['ea'].values[0])

            ase.io.write(self.fpath_processed / f'{split}.xyz', all_structures)
            frame = pd.DataFrame(list(zip(rxns,e_a)), columns=['smiles','ea'])
            frame.to_csv(self.fpath_processed / f'{split}.csv', index=False)

            process_xyz_to_npz((self.fpath_processed / f'{split}.csv').as_posix(),
                               (self.fpath_processed / f'{split}.xyz').as_posix(),
                               (self.fpath_processed / f'{split}.npz').as_posix())
        
        if not self.keep_downloads: shutil.rmtree(self.fpath_download)
             

    def create_folders(self) -> None:
        """
        Creates necessary directories for storing downloaded and processed files if they don't exist.
        """
        
        if not self.fpath_processed.exists(): self.fpath_processed.mkdir(parents=True)
        if not self.fpath_download.exists(): self.fpath_download.mkdir(parents=True)

if __name__ == '__main__':

    data_acquisition = PreProcessTransition1x()
    data_acquisition.start_data_acquisition()
