import argparse
import json
import torch
from prettytable import PrettyTable
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from cgr_mpnn_3D.data.ChemDataset import ChemDataset
from cgr_mpnn_3D.utils.graph_features import process_xyz_to_npz


def activation_energy_prediction(
    input_smiles: str,
    input_coordinates: str = "",
    output_results: str = "",
    model_path: str = "",
    print_results: bool = False,
    store_results: bool = False,
    output_format: str = "text",
) -> None:
    """
    Predict activation energies for chemical reactions based on SMILES and 3D coordinates.

    Args:
        input_smiles (str): Path to the file containing SMILES strings.
        input_coordinates (str): Path to the file containing 3D coordinates.
        output_results (str): Path to save the results.
        model_path (str): Path to the saved model.
        print_results (bool): Whether to print results to the console.
        store_results (bool): Whether to store results in a file.
        output_format (str): Format for stored results ("text" or "json"). Defaults to "text".
    """
    # Resolve input paths
    data_path_smiles = Path(input_smiles)
    data_path_coordinates = Path(input_coordinates)
    data_path_results = Path(output_results) if output_results else Path("results.txt")

    # Ensure proper output file name
    if data_path_results.is_dir():
        data_path_results /= "results.txt"

    # Validate input files
    if not data_path_smiles.is_file():
        raise FileNotFoundError(f"SMILES file not found: {data_path_smiles}")
    if not data_path_coordinates.is_file():
        raise FileNotFoundError(
            f"3D coordinates file not found: {data_path_coordinates}"
        )

    # Convert XYZ to NPZ for processing
    data_path_npz = data_path_coordinates.parent / (data_path_coordinates.stem + ".npz")
    process_xyz_to_npz(data_path_smiles, data_path_coordinates, data_path_npz)

    # Load dataset and model
    pred_data = ChemDataset(
        data_path_smiles.as_posix(), data_npz_path=data_path_npz.as_posix()
    )
    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Prepare table and results list
    table = PrettyTable()
    table.field_names = ["Reaction ID", "Activation Energy [kcal/mol]"]
    results = []

    # Run predictions
    with torch.no_grad():
        for i, data in enumerate(
            tqdm(pred_data, desc="Calculating Activation Energies", dynamic_ncols=True)
        ):
            data.to(device)
            activation_energy = model(data).item()
            results.append(
                {"Reaction_ID": i + 1, "Activation Energy": activation_energy}
            )
            table.add_row([i + 1, f"{activation_energy:.3f}"])

    # Output results
    if print_results:
        print("\nPredicted Activation Energies:\n")
        print(table)

    if store_results:
        if output_format == "text":
            with open(data_path_results, "w") as f:
                f.write("Predicted Activation Energies:\n\n")
                f.write(str(table))
        elif output_format == "json":
            with open(data_path_results.with_suffix(".json"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            raise ValueError("Unsupported output format. Use 'text' or 'json'.")

        print(f"\nResults saved to: {data_path_results}")


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="CLI tool for predicting the activation energy of chemical reactions"
        + " via the CGR MPNN 3D Graph Neural Network."
    )
    args.add_argument(
        "--data_path_smiles",
        default="cli_tool/files/demo.csv",
        type=str,
        help="Path to .csv smiles data set.",
    )
    args.add_argument(
        "--data_path_coordinates",
        default="cli_tool/files/demo.xyz",
        type=str,
        help="Path to .xyz coordinates data set.",
    )
    args.add_argument(
        "--data_path_model",
        default="cli_tool/files/CGR-MPNN-3D_94owmnhg.pth",
        type=str,
        help="Path to .xyz coordinates data set.",
    )
    args.add_argument(
        "--data_path_results",
        default="cli_tool/results.txt",
        type=str,
        help="Path where results should be saved..",
    )
    args.add_argument(
        "--data_path", default="datasets", type=str, help="Path to .csv data sets"
    )
    args.add_argument(
        "--store_results",
        action="store_true",
        help="Flag to save results",
    )
    args.add_argument(
        "--print_results",
        action="store_true",
        help="Flag to print results",
    )

    args = args.parse_args()

    activation_energy_prediction(
        args.data_path_smiles,
        args.data_path_coordinates,
        args.data_path_results,
        args.data_path_model,
        args.print_results,
        args.store_results,
    )
