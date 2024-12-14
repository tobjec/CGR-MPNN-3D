# CGR-MPNN-3D

## Overview

CGR-MPNN-3D is a machine learning framework designed to predict activation energies for chemical reactions. The model integrates a Condensed Graph of Reaction (CGR) Message Passing Neural Network (MPNN) with additional features derived from 3D molecular fingerprints using the MACE framework. By combining 2D and 3D representations of chemical reactions, the model aims to achieve high accuracy in energy predictions.

## Background

The project builds upon the Condensed Graph of Reaction (CGR) framework described in:

> *Heid, Esther, and William H. Green. ["Machine learning of reaction properties via learned representations of the condensed graph of reaction."](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00975) Journal of Chemical Information and Modeling 62.9 (2021): 2101-2110.*

For the 3D features, molecular fingerprints derived from the MACE force field are incorporated:

> *Batatia, Ilyes, et al. ["MACE: Higher order equivariant message passing neural networks for fast and accurate force fields."](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html) Advances in Neural Information Processing Systems 35 (2022): 11423-11436.*

This approach uses the Transition1x (T1x) dataset, which contains reactants, products, and transition states:

> *Schreiner, Mathias, et al. ["Transition1x-a dataset for building generalizable reactive machine learning potentials."](https://www.nature.com/articles/s41597-022-01870-w) Scientific Data 9.1 (2022): 779.*

## Installation
To set up the CGR-MPNN-3D package, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tobjec/CGR-MPNN-3D.git
   cd CGR-MPNN-3D
2. Set up a Python environment and activate it:
   ```bash
   python3 -m venv env
   source env/bin/activate
3. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
4. Ensure CUDA is properly configured (if using GPU acceleration - strongly recommended).

## Usage

### Training

To train the model, use the following command-line interface (CLI):
```bash
python train.py \
  --name CGR-MPNN-3D \
  --depth 4 \
  --hidden_sizes 400 400 400 \
  --dropout_ps 0.1 0.1 0.1 \
  --activation_fn ReLU \
  --save_path saved_models \
  --learnable_skip True \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --weight_decay 1e-5 \
  --batch_size 64 \
  --gamma 0.9 \
  --data_path datasets \
  --gpu_id 0
  --use_logger False
```

#### CLI Arguments for `train.py`

- `--name` (str): Model name (`CGR` or `CGR-MPNN-3D`).
- `--depth` (int): Depth of the GNN (default: ``3).
- `--hidden_sizes` (list of int): Hidden layer sizes (default: `[300, 300, 300]`).
- `--dropout_ps` (list of float): Dropout probabilities (default: `[0.02, 0.02, 0.02]`).
- `--activation_fn` (str): Activation function, choose from ReLU, SiLU, or GELU (default: `ReLU`).
- `--save_path` (str): Path to save trained model parameters (default: `saved_models`).
- `--learnable_skip` (bool): Use of learnable skip connections, True or False (default: `False`).
- `--learning_rate` (float): Learning rate for the optimizer (default: `1e-3`).
- `--num_epochs` (int): Number of training epochs (default: `30`).
- `--weight_decay` (float): Weight decay regularization (default: `0`).
- `--batch_size` (int): Batch size for training (default: `32`).
- `--gamma` (float): Learning rate decay factor (default: `1`).
- `--data_path` (str): Path to dataset directory (default: `datasets`).
- `--gpu_id` (int): Index of GPU to use (default: `0`).
- `--file_path` (str): Path to save training outcomes (default: `parameter_study.json`).
  `--use_logger` (str): Whether to use a WandB logger or not (default: `False`).

#### Example Output
The required datasets will be downloaded and processed automatically if not available. Training results will be logged, and the trained model will be saved in the specified `--save_path` directory. Hyperparameter metadata and test results will be dumped into a JSON file.

### Testing
To test the model, use the following CLI:
```bash
python test.py \
  --path_trained_model saved_models/CGR-MPNN-3D_model.pt \
  --data_path datasets \
  --save_result True \
  --gpu_id 0
```
#### CLI Arguments for `test.py`
- `--path_trained_model` (str): Path to trained model to be tested.
- `--data_path` (str): Base directory for datasets (default: `datasets`).
- `--save_result` (bool): Flag to save test result (default: `False`).
- `--gpu_id` (int): GPU ID to use for testing (default: `0`).

#### Example Output

The test loss (RMSE) will be printed, and results can be optionally saved in a JSON file if `--save_result` is set to `True`.

## Error Metric

The error metric used for this project is the Mean Squared Error (MSE) between predicted and true activation energies.

#### Target Metric

The target for this metric was an MSE of **12.3 kcal/mol**, which was determined based on the benchmark reference model without using the 3D descriptors.

#### Achieved Metric

The achieved MAE after training for the 3D-enhanced model was **8.6 kcal/mol**, surpassing the target value.

## Work Breakdown and Time Tracking

Below is the time spent on each task:

| Task | Planned Time (h) | Actual Time (h) |
|----------|----------|----------|
| Literature Review   | 8   | 12  |
| Dataset Preprocessing | 5   | 10  |
| Pipeline Implementation | 15 | 22 |
| MACE Integration | 8 | 10 |
| Training and Evaluation | 15 | 8 |
| Documentation | 4 | 7 |
|**Total** | 55 | 69 |

## Deviation from Submitted Project Plan

Due to some weird error regarding the MACE finetuning CLI the force field model couldn't be finetuned to the training data from T1x dataset yet. To be continued.




