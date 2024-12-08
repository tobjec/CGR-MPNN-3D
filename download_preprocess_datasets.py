import urllib.request
import numpy as np
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

def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        print(f"\rDownloading: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="")

def process_rxn_files(base_folder: Path, rxn_range: int) -> tuple:
    """
    Process .log files in the specified rxn range and generate corresponding .xyz files.

    Args:
        base_folder (Path): Path to the folder containing rxn directories.
        rxn_range (int): Range of rxn indices to process.
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

def process_log_to_xyz(log_file: Path, xyz_file: Path):
    """
    Extract atomic data from a .log file and write to a .xyz file.

    Args:
        log_file (Path): Path to the .log file.
        xyz_file (Path): Path to the .xyz file.
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


class PreProcessTransition1x:

    def __init__(self, fpath_download: str='downloaded_datasets', fpath_processed: str='datasets_test',
                 dlink_transition: str='https://figshare.com/ndownloader/files/36035789/transition1x.h5',
                 dlink_wb97xd3_csv: str='https://zenodo.org/records/3715478/files/wb97xd3.csv',
                 dlink_wb97xd3: str='https://zenodo.org/records/3715478/files/wb97xd3.tar.gz',
                 rxn_range: int=11961, keep_downloads: bool=False):
        
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
        self.create_folders()
        self.download_files()
        print(f'Start of the extraction of {self.folder_to_extract}')
        self.extract_files()
        print('End of the extraction')
        print('Beginning of the dataset creation.')
        self.create_dataset_files(data_sets)

    def download_files(self):
        
        for url in [self.dlink_transition, self.dlink_wb97xd3, self.dlink_wb97xd3_csv]:
            
            filename = os.path.basename(url)
            file_path = self.fpath_download / filename

            if not file_path.exists():
                file_path.touch()
                print(f'Downloading: {filename}')
                urllib.request.urlretrieve(url, file_path.as_posix(), reporthook=progress_callback)
                print(f'Finished downloading: {filename}', end='\n\n')
        
    def extract_files(self):

        filenames = [os.path.basename(file) for file in [os.path.basename(self.dlink_wb97xd3)]]
        for filename in filenames:
            fpath = self.fpath_download / filename
            folder_to_extract = self.fpath_download
            shutil.unpack_archive(fpath, folder_to_extract)
            print(f'File {filename} extracted to {folder_to_extract.as_posix()}')
    
    def create_dataset_files(self, data_sets: list):
        
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
        
        if not self.keep_downloads: shutil.rmtree(self.fpath_download)
             

    def create_folders(self):
        
        if not self.fpath_processed.exists(): self.fpath_processed.mkdir(parents=True)
        if not self.fpath_download.exists(): self.fpath_download.mkdir(parents=True)

if __name__ == '__main__':

    data_acquisition = PreProcessTransition1x()
    data_acquisition.start_data_acquisition()
