import wandb
import argparse
from wandb_logger import WandBLogger
import torch.nn.functional as F
from train import train
import json

def train_with_sweeps():
    """
    Train a model using hyperparameter sweeps with Weights & Biases (wandb).

    This function initializes a wandb run, retrieves the sweep configuration, 
    and trains a model based on the provided hyperparameters. Metrics are logged 
    during training and the run is finalized at the end.
    """
    wandb.init(
        entity="tobjec",
        project="CGR-MPNN-3D",
        group="tuw"
    )

    # Retrieve the configuration from wandb
    config = wandb.config

    # Train the model with hyperparameters from the sweep configuration
    train_result = train(
        name=f"{config.name}_{wandb.run.id}",
        depth=config.depth,
        hidden_sizes=config.hidden_sizes * config.depth,
        dropout_ps=config.dropout_ps * config.depth,
        activation_fn=F.relu,  # Activation function
        save_path=config.save_path,
        use_learnable_skip=config.learnable_skip,
        lr=config.lr,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        gamma=config.gamma,
        data_path=config.data_path,
        gpu_id=config.gpu_id
    )

    # Extract training and validation losses from the training results
    train_losses = train_result.get("train_losses", [])
    val_losses = train_result.get("val_losses", [])

    # Log training and validation losses epoch by epoch
    for epoch, train_loss in enumerate(train_losses):
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss
        })

        # Log validation loss every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == len(train_losses) - 1:
            val_loss_index = epoch // 5 if epoch % 5 == 0 else len(val_losses) - 1
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_losses[val_loss_index]
            })

    # Finalize the wandb run
    wandb.finish()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='CLI for conducting hyperparameter tuning')
    args.add_argument('-p', '--path_input_file', default='hyperparameter_study/sweep_config.json',
                      help='Path to the config file')
    args = args.parse_args()

    with open(args.path_input_file, "r") as f:
        sweep_config = json.load(f)

    sweep_id = wandb.sweep(sweep_config, project="CGR-MPNN-3D", entity="tobjec")

    wandb.agent(sweep_id, function=train_with_sweeps, count=20)