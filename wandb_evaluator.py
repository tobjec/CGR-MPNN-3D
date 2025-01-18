import argparse
import wandb
import json

def evaluate_sweep(entity_name: str, project_name: str, sweep_id: str, output_file: str=None) -> None:
    """
    Evaluate a W&B sweep and retrieve all configurations, 
    training and validation losses of the runs.

    Args:
        entity_name (str): Name of the W&B entity.
        project_name (str): Name of the W&B project.
        sweep_id (str): ID of the W&B sweep.
        output_file (str): Path to save the evaluation results in a JSON file.

    Returns:
        None
    """

    # Initialize API
    api = wandb.Api()

    # Get sweep object
    sweep = api.sweep(f"{entity_name}/{project_name}/{sweep_id}")

    # Access all runs in the sweep
    runs = sweep.runs

    # Store results
    results = []

    for run in runs:
        result = {
            "run_id": run.id,
            "train_loss": run.summary.get("train_loss"),  # Replace key if named differently
            "val_loss": run.summary.get("val_loss"),      # Replace key if named differently
            "config": run.config,
        }
        results.append(result)

    # Sort by validation loss
    sorted_results = sorted(results, key=lambda x: x["val_loss"] if x["val_loss"] is not None else float("inf"))

    # Get the best run
    best_run = sorted_results[0] if sorted_results else None

    # Print the results
    print("\nSweep Evaluation Results:")
    for result in sorted_results:
        print(f"Run ID: {result['run_id']}, Train Loss: {result['train_loss']}, Val Loss: {result['val_loss']}")
        print(f"Configuration: {result['config']}")
        print("-" * 50)

    if best_run:
        print("\nBest Run:")
        print(f"Run ID: {best_run['run_id']}, Train Loss: {best_run['train_loss']}, Val Loss: {best_run['val_loss']}")
        print(f"Configuration: {best_run['config']}")

    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(sorted_results, f, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate W&B Sweep")
    parser.add_argument("--entity_name", type=str, required=True, help="Name of the W&B entity")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the W&B project")
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the W&B sweep")
    parser.add_argument(
        "--output_file", type=str, default="sweep_results.json", help="Path to save the results JSON file"
    )

    args = parser.parse_args()

    evaluate_sweep(args.entity_name, args.project_name, args.sweep_id, output_file=args.output_file)