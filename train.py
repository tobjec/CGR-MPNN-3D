import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import os

from cgr_mpnn_3D.models import CGR
from cgr_mpnn_3D.data import ChemDataset
from cgr_mpnn_3D.training.trainer import RxnGraphTrainer
from cgr_mpnn_3D.utils.json_dumper import json_dumper
from download_preprocess_datasets import PreProcessTransition1x


def train(name: str, depth: int=3, hidden_sizes: list=[300,300,300], dropout_ps: list=[0.02,0.02,0.02],
          activation_fn: F=F.relu, save_path: str='saved_models', use_learnable_skip: bool=False,
          lr: float=1e-3, num_epochs: int=30, weight_decay: float=0, batch_size: int=32,
          gamma: float=1, data_path: str='datasets', gpu_id: int=0) -> dict:

    data_path_train = Path(data_path) / 'train.csv'
    data_path_val = Path(data_path) / 'val.csv'

    data_sets = []
    if not data_path_train.exists(): data_sets.append('train')
    if not data_path_val.exists(): data_sets.append('val')
    if data_sets: PreProcessTransition1x().start_data_acquisition(data_sets)

    train_data = ChemDataset(data_path_train)
    val_data = ChemDataset(data_path_val)
    
    match name:
        case 'CGR':
            model = CGR(train_data[0].num_node_features, train_data[0].num_edge_features,
                        depth=depth, hidden_sizes=hidden_sizes, dropout_ps=dropout_ps,
                        activation_fn=activation_fn, use_learnable_skip=use_learnable_skip)
        case 'CGR_MPNN_3D':
            pass
        case _:
            raise NameError(f'Unkown model with name {name}')

        
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    loss_fn = torch.nn.MSELoss()

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    trainer = RxnGraphTrainer(name,
                    model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_data,
                    val_data,
                    device,
                    num_epochs, 
                    save_path,
                    batch_size=batch_size)
    
    return trainer.train()


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='CLI tool for training the CGR MPNN 3D Graph Neural Network.')
    args.add_argument('-n', '--name', default='CGR', choices=['CGR', 'CGR_MPNN_3D'], type=str,
                      help='Type of the model to be trained')
    args.add_argument('-d', '--depth', default=3, type=int, help='Depth of GNN')
    args.add_argument('-h', '--hidden_sizes', default=[300, 300, 300], nargs='+', type=int,
                      help='Size of hidden layers')
    args.add_argument('-p', '--dropout_ps', default=[0.02, 0.02, 0.02], nargs='+', type=float,
                      help='Dropout probability of the hidden layers')
    args.add_argument('-a', '--activation_fn', default='ReLU', choices=['ReLU', 'SiLU', 'GELU'], type=str,
                      help='Activation function for the GNN')
    args.add_argument('-s', '--save_path', default='saved_models', type=str, help='Path to the saved model parameters')
    args.add_argument('-k', '--learnable_skip', default='False', type=str, help='Using of learnable skip connections')
    args.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='Learning rate for the GNN')
    args.add_argument('-e', '--num_epochs', default=30, type=int, help='Number of training epochs')
    args.add_argument('-w', '--weight_decay', default=0, type=float, help='Weight decay regularization for the optimizer')
    args.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size of the training data')
    args.add_argument('-m', '--gamma', default=1, type=float, help='Gamma value for the learning rate scheduler')
    args.add_argument('-u', '--data_path', default='datasets', type=str, help='Path to .csv data sets')
    args.add_argument('-g', '--gpu_id', default=0, type=int, help='Index of which GPU to use')
    args.add_argument('-f', '--file_path', default='parameter_study.json', type=str, help='Filename to training outcomes')
    
    args = args.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    name = '_'.join([args.name, f'd-{args.depth}','h-'+'-'.join([str(i) for i in args.hidden_sizes]), 
                    'p-'+'-'.join([str(i) for i in args.dropout_ps]), args.activation_fn,
                    f's-{'t' if args.learnable_skip=='True' else 'f'}', f'l-{args.learning_rate}',
                    f'e-{args.num_epochs}', f'w-{args.weight_decay}', f'b-{args.batch_size}',
                    f'g-{args.gamma}'])

    result_metadata_dict = {name: {'metadata': {'depth': args.depth, 'hidden_sizes': args.hidden_sizes,
                                            'dropout_ps': args.dropout_ps,'activation_fn': args.activation_fn,
                                            'learnable_skip': args.learnable_skip, 'lr': args.learning_rate,
                                            'num_epochs': args.num_epochs, 'weight_decay': args.weight_decay,
                                            'batch_size': args.batch_size, 'gamma': args.gamma}}}

    match args.activation_fn:
        case 'ReLu':
            args.activation_fn = F.relu
        case 'SiLU':
            args.activation_fn = F.silu
        case 'GELU':
            args.activation_fn = F.gelu
    
    args.learnable_skip = False if args.learnable_skip=='False' else True

    train_result = train(name, args.depth, args.hidden_sizes, args.dropout_ps, args.activation_fn,
                         args.save_path, args.learnable_skip, args.learnable_skip, args.learning_rate,
                         args.num_epochs, args.weight_decay, args.batch_size, args.gamma, args.data_path, args.gpu_id)
    
    result_metadata_dict[name].update(**train_result)
        
    
    json_file_path = Path('hyperparameter_study').mkdir(parents=True, exist_ok=True) / \
                     f'{args.name}_hyperparameter_study.json'
    
    json_dumper(json_file_path.as_posix(), result_metadata_dict)
