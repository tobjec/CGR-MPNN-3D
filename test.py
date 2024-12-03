## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
from pathlib import Path

from torchvision.models import resnet18 # change to the model you want to test
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

# Path to the CIFAR10 Dataset
path_cifar10 = "/home/tjechtl/Downloads/cifar-10-batches-py/"

transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

def test(path_trained_model, modelIn, transformerIn, path_cifar10=path_cifar10):
    
    transform = transformerIn
    
    test_data = CIFAR10Dataset(path_cifar10, Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_test_data = len(test_data)

    model = modelIn

    model = DeepClassifier(model)
    model.load(path_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss 
    ### and metrics to terminal after testing is finished
    with torch.no_grad():
        n_samples = len(test_data_loader)
        total_loss = 0.0
        test_metric.reset()

        for i, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            total_loss += loss.item()
            test_metric.update(predictions, labels)
        
        mean_loss = total_loss / n_samples
        mean_accuracy, mean_per_class_accuracy = test_metric.accuracy(), test_metric.per_class_accuracy()
        print(f"Test loss: {mean_loss:.4f}\n"+str(test_metric))
    
    test_dict = {'test_losses': mean_loss, 'test_mean_a': mean_accuracy, 'test_mean_pca': mean_per_class_accuracy}

    return test_dict

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='5', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    print(os.getcwd()+'\n\n')
    test("saved_models/ResNet18-0-3-standard.pth", resnet18(weights=None),
         transform)