## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
from tqdm import tqdm
import json
from concurrent import futures

from dlvc.models.class_model import DeepClassifier # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from test import test

from dlvc.models.cnn import MyCNN

modelIn = MyCNN()

nameIn = 'CNN'

num_epoch = 30

# Path to the CIFAR10 Dataset
path_cifar10 = "/home/tjechtl/Downloads/cifar-10-batches-py/"

optimizer = torch.optim.AdamW

transformer1 = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

transformer2 = v2.Compose([v2.ToImage(), v2.RandomVerticalFlip(), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

transformer3 = v2.Compose([v2.ToImage(), v2.RandomCrop(32, padding=4), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

# Naming convention is the model name, the weight decay exponent, the exponent of learning rate and transformer
names = [f"{nameIn}-0-3-standard", f"{nameIn}-0-3-vertical_flip", f"{nameIn}-4-3-crop", f"{nameIn}-2-2-standard"]

weight_decays = [0,0,1e-4,1e-2]

learning_rates = [1e-3, 1e-3, 1e-3, 1e-2]

transformers = [transformer1, transformer2, transformer3, transformer1]


def train(name, num_epochs, transformerIn, optimizerIn, weight_decayIn=0, lrIn=1e-3, path_cifar10=path_cifar10):

    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure 
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation
    
    
    train_transform = transformerIn
    
    val_transform = transformerIn
    
    train_data = CIFAR10Dataset(path_cifar10, Subset.TRAINING, transform=train_transform)
    
    val_data = CIFAR10Dataset(path_cifar10, Subset.VALIDATION, transform=val_transform) 
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = modelIn
    model.train()

    model = DeepClassifier(model)
    model.to(device)
    optimizer = optimizerIn(model.parameters(), lr=lrIn, amsgrad=True, weight_decay=weight_decayIn)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    
    trainer = ImgClassificationTrainer(name,
                    model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    num_epochs, 
                    model_save_dir,
                    batch_size=128, # feel free to change
                    val_frequency = val_frequency)
    
    return trainer.train()

# Function to parallelize the training 
def train_and_test(name, weight_decay, lr, transformer, optimizer, path_cifar10, modelIn):

    train_result = train(name, num_epoch, transformer, optimizer, weight_decayIn=weight_decay, lrIn=lr, path_cifar10=path_cifar10)
    
    test_result = test(Path("saved_models") / (name + '.pth'), modelIn, transformer, path_cifar10=path_cifar10)
    
    train_result.update(test_result)
    return name, train_result


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    dict_parameter_study_resnet = {}

    for name, weight_decay, lr, transformer in zip(names, weight_decays, learning_rates, transformers):
        dict_parameter_study_resnet[name] = train(name, num_epoch, transformer, optimizer, weight_decayIn=weight_decay,
                           lrIn=lr, path_cifar10=path_cifar10)
        
        dict_parameter_study_resnet[name].update(test(Path("saved_models") / (name+'.pth'),
                                                      modelIn, transformer,
                                                      path_cifar10=path_cifar10))
    
    #with futures.ProcessPoolExecutor(max_workers=4) as e:
    #    # Create a list of futures
    #    futures = [e.submit(train_and_test, name, weight_decay, lr, transformer, optimizer, path_cifar10, modelIn)
    #            for name, weight_decay, lr, transformer in zip(names, weight_decays, learning_rates, transformers)]
    #    
    #    # Collect results as they complete
    #    for future in futures.as_completed(futures):
    #        name, result = future.result()
    #        dict_parameter_study_resnet[name] = result
        
    
    file_path = f'{nameIn}_parameter_study.json'

    # Check if the file already exists and has content
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    data.update(dict_parameter_study_resnet)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
