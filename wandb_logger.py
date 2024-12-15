import torch
import wandb


class WandBLogger:
    """
    A utility class for logging experiment metrics and model details to Weights & Biases (wandb).

    Args:
        enabled (bool, optional): Whether to enable logging to wandb. Defaults to True.
        model (torch.nn.Module, optional): PyTorch model to monitor with wandb. Defaults to None.
        run_name (str, optional): Custom name for the wandb run. If None, defaults to the run ID. Defaults to None.
        config (dict, optional): Configuration dictionary for wandb initialization. Defaults to None.
    """

    def __init__(
        self,
        enabled=True,
        model: torch.nn.Module = None,
        run_name: str = None,
        config: dict = None,
    ):
        self.enabled = enabled  # Whether wandb logging is enabled
        self.config = config or {}  # Configuration dictionary

        if self.enabled:
            wandb.init(
                entity="tobjec", project="CGR-MPNN-3D", group="tuw", config=self.config
            )

            # Set the run name to the provided name or default to the run ID
            if run_name is None:
                wandb.run.name = wandb.run.id
            else:
                wandb.run.name = run_name

            # Optionally watch the model
            if model is not None:
                self.watch(model)

    def watch(self, model: torch.nn.Module, log_freq: int = 1) -> None:
        """
        Monitors a PyTorch model with wandb for logging gradients and parameters.

        Args:
            model (torch.nn.Module): PyTorch model to monitor.
            log_freq (int, optional): Frequency of logging updates. Defaults to 1.
        """
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=True, step=None) -> None:
        """
        Logs metrics and other data to wandb.

        Args:
            log_dict (dict): Dictionary of metrics or data to log.
            commit (bool, optional): Whether to mark this as the final step in the current logging context. Defaults to True.
            step (int, optional): Step number to associate with the logged data. Defaults to None.
        """
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self) -> None:
        """
        Ends the wandb run, ensuring that all logs are saved and finalized.
        """
        if self.enabled:
            wandb.finish()
