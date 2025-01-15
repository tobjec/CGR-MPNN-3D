import argparse
import os
import torch
import matplotlib.pyplot as plt
import torch_geometric as tg
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.utils.json_dumper import json_dumper
from download_preprocess_datasets import PreProcessTransition1x

############################## PLOT FORMATTING ################################

plt.rc('figure', autolayout=True)
plt.rc('mathtext', default='regular')
plt.rc('axes', linewidth=1.5)

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

# Use rcParams for tick width and size
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5

plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3

# Activate LaTeX style if supported
plt.rc('text', usetex=False)
plt.rc('font', family='serif')


def test(
    name: str, path_trained_model: str, data_path: str = "datasets", gpu_id: int = 0
) -> dict:
    """
    Test a trained model on a dataset.

    Args:
        name (str): The name of the model architecture (e.g., 'CGR', 'CGR_MPNN_3D').
        path_trained_model (str): Path to the trained model file.
        data_path (str, optional): Base directory for datasets. Defaults to 'datasets'.
        gpu_id (int, optional): GPU ID to use for testing. Defaults to 0.

    Returns:
        dict: A dictionary containing the test loss metrics.
    """

    # Define the path to the test dataset
    data_path_test = Path(data_path) / "test.csv"

    # Check if test dataset exists, otherwise acquire it
    data_sets = []
    if not data_path_test.exists():
        data_sets.append("test")
    else:
        print("Test data set found at", data_path_test)

    if data_sets:
        PreProcessTransition1x().start_data_acquisition(data_sets)

    # Initialize the model based on the name
    match name:
        case "CGR":
            # Load test dataset
            test_data = ChemDataset(data_path_test)
        case "CGR-MPNN-3D":
            # Define the path to the test dataset
            data_path_test_npz = Path(data_path) / "test.npz"
            # Load test dataset
            test_data = ChemDataset(
                data_path_test, data_npz_path=data_path_test_npz.as_posix()
            )
        case _:
            raise NameError(f"Unknown model with name '{name}'.")

    # Initialize data loader for the test set
    test_data_loader = tg.loader.DataLoader(
        test_data,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=torch.cuda.is_available(),
    )

    # Load the trained model    
    model = torch.load(path_trained_model, map_location="cpu")

    # Set up the device for testing
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define the loss function
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Evaluate the model
    with torch.no_grad():
        total_loss = 0.0

        for data in test_data_loader:
            data = data.to(device)
            predictions = model(data)
            # Reverse standardization for predictions before loss computation
            loss = loss_fn(predictions, data.y)
            total_loss += loss.item()

        # Calculate mean test loss (root mean squared error)
        mean_loss = np.sqrt(total_loss / len(test_data_loader.dataset))
        print(f"Test loss: {mean_loss:.4f}\n")

    # Prepare and return the results
    test_dict = {"test_losses": mean_loss}
    return test_dict

def test(
    name: str, path_trained_model: str, data_path: str = "datasets", gpu_id: int = 0, plot_results: bool = True
) -> dict:
    """
    Test a trained model on a dataset and optionally plot predicted vs. true values.

    Args:
        name (str): The name of the model architecture (e.g., 'CGR', 'CGR_MPNN_3D').
        path_trained_model (str): Path to the trained model file.
        data_path (str, optional): Base directory for datasets. Defaults to 'datasets'.
        gpu_id (int, optional): GPU ID to use for testing. Defaults to 0.
        plot_results (bool, optional): Flag to plot predicted vs. true values. Defaults to True.

    Returns:
        dict: A dictionary containing the test loss metrics.
    """

    # Define the path to the test dataset
    data_path_test = Path(data_path) / "test.csv"

    # Check if test dataset exists, otherwise acquire it
    data_sets = []
    if not data_path_test.exists():
        data_sets.append("test")
    else:
        print("Test data set found at", data_path_test)

    if data_sets:
        PreProcessTransition1x().start_data_acquisition(data_sets)

    # Initialize the model based on the name
    match name:
        case "CGR":
            test_data = ChemDataset(data_path_test)
        case "CGR-MPNN-3D":
            data_path_test_npz = Path(data_path) / "test.npz"
            test_data = ChemDataset(
                data_path_test, data_npz_path=data_path_test_npz.as_posix()
            )
        case _:
            raise NameError(f"Unknown model with name '{name}'.")

    # Initialize data loader for the test set
    test_data_loader = tg.loader.DataLoader(
        test_data,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=torch.cuda.is_available(),
    )

    # Load the trained model    
    model = torch.load(path_trained_model, map_location="cpu")

    # Set up the device for testing
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define the loss function
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Prepare storage for predictions and true values
    all_predictions = []
    all_true_values = []

    # Evaluate the model
    with torch.no_grad():
        total_loss = 0.0

        for data in test_data_loader:
            data = data.to(device)
            predictions = model(data).cpu().numpy()
            true_values = data.y.cpu().numpy()

            all_predictions.extend(predictions)
            all_true_values.extend(true_values)

            loss = loss_fn(torch.tensor(predictions), torch.tensor(true_values))
            total_loss += loss.item()

        # Calculate mean test loss (root mean squared error)
        mean_loss = np.sqrt(total_loss / len(test_data_loader.dataset))
        print(f"Test loss: {mean_loss:.4f}\n")

    # Plot predicted vs. true values
    if plot_results:
        plt.figure(figsize=(10, 8))
        plt.scatter(all_true_values, all_predictions, alpha=0.7, label="Predictions")
        plt.plot(
            [min(all_true_values), max(all_true_values)],
            [min(all_true_values), max(all_true_values)],
            color="red",
            linestyle="--",
            label="Identity Line",
        )
        plt.xlabel("True Activation Energies [kcal/mol]", fontsize=16)
        plt.ylabel("Predicted Activation Energies [kcal/mol]", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which='major', axis='both', color='gray', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.savefig("predicted_vs_true_activation_energies.pdf")
        plt.show()

    # Prepare and return the results
    test_dict = {"test_losses": mean_loss}
    return test_dict


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="CLI tool for testing the CGR MPNN 3D Graph Neural Network."
    )
    args.add_argument("--path_trained_model", help="Path to trained model to be tested")
    args.add_argument(
        "--data_path", default="datasets", type=str, help="Path to .csv data sets"
    )
    args.add_argument(
        "--save_result",
        action='store_true',
        help="Flag to save test result",
    )
    args.add_argument(
        "--gpu_id", default="0", type=str, help="Index of which GPU to use"
    )
    args.add_argument(
        "--plot_results",
        action="store_true",
        help="Flag to plot the predicted vs. real results."
    )

    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    name = os.path.basename(args.path_trained_model).split("_")[0]

    if not Path(args.path_trained_model).exists():
        raise NameError(f"Invalid model data location at {args.path_trained_model}")

    test_dict = test(name, args.path_trained_model, args.data_path, args.plot_results)
    if args.save_result:
        json_file_path = Path("hyperparameter_study")
        json_file_path.mkdir(parents=True, exist_ok=True)
        json_file_path /= f"{name}_hyperparameter_study.json"
        json_dumper(json_file_path.as_posix(), test_dict, args.path_trained_model)
