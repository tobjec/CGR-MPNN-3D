import argparse
import os
import torch
import torch_geometric as tg
import numpy as np
from pathlib import Path
import os

from cgr_mpnn_3D.models.CGR import GNN
from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.utils.json_dumper import json_dumper
from cgr_mpnn_3D.utils.standardizer import Standardizer
from download_preprocess_datasets import PreProcessTransition1x


def test(name: str, path_trained_model: str, data_path: str='datasets', gpu_id: int=0):
    
    data_path_test = Path(data_path) / 'test.csv'

    data_sets = []
    if not data_path_test.exists(): data_sets.append('test')
    else: print('Test data set found at', data_path_test)
    if data_sets: PreProcessTransition1x().start_data_acquisition(data_sets)

    test_data = ChemDataset(data_path_test)

    test_data_loader = tg.loader.DataLoader(test_data, shuffle=False, num_workers=os.cpu_count()//2, 
                                                   pin_memory=torch.cuda.is_available())

    stdizer = Standardizer(test_data_loader)

    match name:
        case 'CGR':
            model = GNN(test_data[0].num_node_features, test_data[0].num_edge_features)
        case 'CGR_MPNN_3D':
            pass
        case _:
            raise NameError(f'Unknown model with name {name}')
    
    state_dict = torch.load(path_trained_model,weights_only=True)

    model.load_state_dict(state_dict)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction='sum')

    with torch.no_grad():
        n_samples = len(test_data_loader)
        total_loss = 0.0

        for data in test_data_loader:
            data = data.to(device)
            predictions = model(data)
            loss = loss_fn(stdizer(predictions, rev=True), data.y)

            total_loss += loss.item()
        
        mean_loss = np.sqrt(total_loss / n_samples)
        print(f"Test loss: {mean_loss:.4f}\n")
    
    test_dict = {'test_losses': mean_loss}

    return test_dict

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='CLI tool for testing the CGR MPNN 3D Graph Neural Network.')
    args.add_argument('-pm', '--path_trained_model', help='Path to trained model to be tested')
    args.add_argument('-dp', '--data_path', default='datasets', type=str, help='Path to .csv data sets')
    args.add_argument('-sr', '--save_result', default='True', choices=['True', 'False'], type=str, help='Flag to save test result')
    args.add_argument('-gid', '--gpu_id', default='0', type=str,
                      help='Index of which GPU to use')
    
    args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.save_result = True if args.save_result=='True' else False

    name = os.path.basename(args.path_trained_model).split('_')[0]

    if not Path(args.path_trained_model).exists(): raise NameError(f'Invalid model data location at {args.path_trained_model}')
    test_dict = test(name, args.path_trained_model, args.data_path)
    if args.save_result: 
        json_file_path = Path('hyperparameter_study')
        json_file_path.mkdir(parents=True, exist_ok=True)
        json_file_path /= f'{name}_hyperparameter_study.json'
        json_dumper(json_file_path.as_posix(), test_dict, args.path_trained_model)