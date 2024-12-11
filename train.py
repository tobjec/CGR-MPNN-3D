import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import os

from cgr_mpnn_3D.models.CGR import GNN
from cgr_mpnn_3D.models.CGR_MPNN_3D import GNN2
from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.training.trainer import RxnGraphTrainer
from cgr_mpnn_3D.utils.json_dumper import json_dumper
from download_preprocess_datasets import PreProcessTransition1x
from wandb_logger import WandBLogger
from test import test


def train(name: str, 
          depth: int = 3, 
          hidden_sizes: list = [300, 300, 300], 
          dropout_ps: list = [0.02, 0.02, 0.02],
          activation_fn: F = F.relu, 
          save_path: str = 'saved_models', 
          use_learnable_skip: bool = False,
          lr: float = 1e-3, 
          num_epochs: int = 30, 
          weight_decay: float = 0, 
          batch_size: int = 32,
          gamma: float = 1, 
          data_path: str = 'datasets', 
          gpu_id: int = 0, 
          logger: WandBLogger = None) -> dict:
    """
    Train a specified model on the training dataset and validate on a validation dataset.

    Args:
        name (str): Name of the model (e.g., 'CGR', 'CGR_MPNN_3D').
        depth (int): Number of layers for the model.
        hidden_sizes (list): List of hidden layer sizes.
        dropout_ps (list): List of dropout probabilities.
        activation_fn (callable): Activation function (default: ReLU).
        save_path (str): Path to save the trained model.
        use_learnable_skip (bool): Whether to use learnable skip connections.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        weight_decay (float): Weight decay for the optimizer.
        batch_size (int): Batch size for training and validation.
        gamma (float): Learning rate decay factor.
        data_path (str): Base directory for datasets.
        gpu_id (int): GPU ID for training (default: 0).
        logger (WandBLogger, optional): WandB logger instance for logging.

    Returns:
        dict: A dictionary containing training results.
    """

    # Define paths to the training and validation datasets
    data_path_train = Path(data_path) / 'train.csv'
    data_path_val = Path(data_path) / 'val.csv'

    # Check for the presence of datasets
    data_sets = []
    if not data_path_train.exists():
        data_sets.append('train')
    else:
        print('Train data set found at', data_path_train)

    if not data_path_val.exists():
        data_sets.append('val')
    else:
        print('Validation data set found at', data_path_val)

    if data_sets:
        PreProcessTransition1x().start_data_acquisition(data_sets)

    # Load the training and validation datasets
    train_data = ChemDataset(data_path_train.as_posix())
    val_data = ChemDataset(data_path_val.as_posix())

    # Initialize the model based on the name
    match name.split('_')[0]:
        case 'CGR':
            model = GNN(
                train_data[0].num_node_features,
                train_data[0].num_edge_features,
                depth=depth,
                hidden_sizes=hidden_sizes,
                dropout_ps=dropout_ps,
                activation_fn=activation_fn,
                use_learnable_skip=use_learnable_skip
            )
        case 'CGR_MPNN_3D':
            model = GNN2(
                train_data[0].num_node_features,
                train_data[0].num_edge_features
            )
        case _:
            raise NameError(f"Unknown model with name '{name}'.")

    # Set up the device for training
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'Starting training on CUDA:{gpu_id}.')
    else:
        print('Starting training on CPU.')
    model.to(device)

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Initialize the trainer
    trainer = RxnGraphTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path,
        batch_size=batch_size,
        logger=logger
    )

    # Start training and return the results
    return trainer.train()



if __name__ == "__main__":

    args = argparse.ArgumentParser(description='CLI tool for training the CGR MPNN 3D Graph Neural Network.')
    args.add_argument('-n', '--name', default='CGR', choices=['CGR', 'CGR_MPNN_3D'], type=str,
                      help='Type of the model to be trained')
    args.add_argument('-d', '--depth', default=3, type=int, help='Depth of GNN')
    args.add_argument('-hs', '--hidden_sizes', default=[300, 300, 300], nargs='+', type=int,
                      help='Size of hidden layers')
    args.add_argument('-ds', '--dropout_ps', default=[0.02, 0.02, 0.02], nargs='+', type=float,
                      help='Dropout probability of the hidden layers')
    args.add_argument('-af', '--activation_fn', default='ReLU', choices=['ReLU', 'SiLU', 'GELU'], type=str,
                      help='Activation function for the GNN')
    args.add_argument('-sp', '--save_path', default='saved_models', type=str, help='Path to the saved model parameters')
    args.add_argument('-ls', '--learnable_skip', default='False', choices=['True', 'False'],
                      type=str, help='Using of learnable skip connections')
    args.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate for the GNN')
    args.add_argument('-ne', '--num_epochs', default=30, type=int, help='Number of training epochs')
    args.add_argument('-wd', '--weight_decay', default=0, type=float, help='Weight decay regularization for the optimizer')
    args.add_argument('-bs', '--batch_size', default=32, type=int, help='Batch size of the training data')
    args.add_argument('-g', '--gamma', default=1, type=float, help='Gamma value for the learning rate scheduler')
    args.add_argument('-dp', '--data_path', default='datasets', type=str, help='Path to .csv data sets')
    args.add_argument('-gid', '--gpu_id', default=0, type=int, help='Index of which GPU to use')
    args.add_argument('-fp', '--file_path', default='parameter_study.json', type=str, help='Filename to training outcomes')
    
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
    
    wandb_config = result_metadata_dict[name]['metadata']
    logger = WandBLogger(config=wandb_config)

    print('Metadata of the training:')
    for key,value in wandb_config.items():
        print(f'{key}: {value}')

    match args.activation_fn:
        case 'ReLU':
            args.activation_fn = F.relu
        case 'SiLU':
            args.activation_fn = F.silu
        case 'GELU':
            args.activation_fn = F.gelu
        case _:
            raise NameError(f'Unknown activation function {args.activation_fn}.')
    
    args.learnable_skip = False if args.learnable_skip=='False' else True

    train_result = train(name, args.depth, args.hidden_sizes, args.dropout_ps, args.activation_fn, args.save_path,
                         args.learnable_skip, args.learning_rate, args.num_epochs, args.weight_decay,
                         args.batch_size, args.gamma, args.data_path, args.gpu_id, logger)
    test_result = test(f"{args.save_path}/{name}.pth")
    
    result_metadata_dict[name].update(**train_result)
    result_metadata_dict[name].update(**test_result)
       
    json_file_path = Path('hyperparameter_study')
    json_file_path.mkdir(parents=True, exist_ok=True)
    json_file_path /= f'{args.name}_hyperparameter_study.json'
    
    json_dumper(json_file_path.as_posix(), result_metadata_dict)
