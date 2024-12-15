import argparse
import os
import torch
import torch_geometric as tg
import numpy as np
from pathlib import Path
import os

from cgr_mpnn_3D.models.GNN import GNN
from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.utils.json_dumper import json_dumper
from cgr_mpnn_3D.utils.standardizer import Standardizer
from download_preprocess_datasets import PreProcessTransition1x


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

    # Initialize standardizer for the dataset
    stdizer = Standardizer(test_data_loader)

    model = GNN(test_data[0].num_node_features, test_data[0].num_edge_features)

    # Load the trained model's weights
    state_dict = torch.load(path_trained_model, map_location="cpu")
    model.load_state_dict(state_dict)

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
            loss = loss_fn(stdizer(predictions, rev=True), data.y)
            total_loss += loss.item()

        # Calculate mean test loss (root mean squared error)
        mean_loss = np.sqrt(total_loss / len(test_data_loader.dataset))
        print(f"Test loss: {mean_loss:.4f}\n")

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
        default="False",
        choices=["True", "False"],
        type=str,
        help="Flag to save test result",
    )
    args.add_argument(
        "--gpu_id", default="0", type=str, help="Index of which GPU to use"
    )

    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.save_result = True if args.save_result == "True" else False

    name = os.path.basename(args.path_trained_model).split("_")[0]

    if not Path(args.path_trained_model).exists():
        raise NameError(f"Invalid model data location at {args.path_trained_model}")

    test_dict = test(name, args.path_trained_model, args.data_path)
    if args.save_result:
        json_file_path = Path("hyperparameter_study")
        json_file_path.mkdir(parents=True, exist_ok=True)
        json_file_path /= f"{name}_hyperparameter_study.json"
        json_dumper(json_file_path.as_posix(), test_dict, args.path_trained_model)
