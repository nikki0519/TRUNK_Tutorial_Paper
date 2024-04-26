# ----------------------------------------------------
# Name: metrics.py
# Purpose: Script to train and test resnet quantized
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
from torchvision.models import resnet50
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qconfig, prepare_qat
import torch.nn as nn
import torch
import numpy as np
import copy
from metrics import *

# Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device}")
device = torch.device(device)

loss_function = nn.CrossEntropyLoss() # loss function to compare target with prediction

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load the model checkpoint which is saved at every epoch

    Parameters
    ----------
    model: torchvision.models
        the chosen model by the user

    optimizer: torch.optim
        propagation function

    checkpoint_path: str
        path to model checkpoints
    """

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    loss_per_iteration = checkpoint.get('loss_per_iteration', [])  # Get previous loss history if available
    return start_epoch, loss_per_iteration

@run_time
def train(model, trainloader, epochs, learning_rate, betas, weight_decay, save_to_path=None):
    """
    Train the model on the chosen dataset

    Parameters
    ----------
    model: torchvision.models
        the chosen model by the user

    trainloader: torch.utils.data.DataLoader
        iterable dataset 

    epochs: int
        number of epochs to train over

    learning_rate: float
        learning rate for optimizer

    weight_decay: float
        weight decay for optimizer

    save_to_path: str [Optional]
        path to save the model

    Return
    ------
    loss_per_iteration: list
        list of training loss over all the epochs and batches
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    model = model.to(device)
    model.train()
    loss_per_iteration = []
    start_epoch = 1  # Default start epoch

    if(os.path.exists(save_to_path)):
        start_epoch, loss_per_iteration = load_checkpoint(model, optimizer, save_to_path)

    for epoch in range(start_epoch, epochs + 1):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if((batch_idx + 1) % 100 == 0):
                print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch, batch_idx + 1, running_loss / 100))
                loss_per_iteration.append(running_loss / 100)
                running_loss = 0.0

        if(save_to_path):
            # Also save the optimizer state and current epoch for resuming training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_per_iteration': loss_per_iteration,
            }, save_to_path)

    return loss_per_iteration

@run_time
def test(model, path_to_network, testloader, num_classes):
    """
    Conduct validation on the model for the chosen dataset

    Parameters
    ----------
    model: torchvision.models
        the chosen model by the user

    path_to_network: str
        path where model is saved

    testloader: torch.utils.data.DataLoader
        iterable dataset 

    num_classes: int
        number of classes

    Return
    ------
    confusion_matrix: np.array
        the confusion matrix for the dataset using this trained model

    accuracy: float
        the accuracy of the model
    """

    checkpoint = torch.load(path_to_network)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.device("cpu:0"))
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(torch.device("cpu:0"))
            labels = labels.to(torch.device("cpu:0"))
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1
            
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy

class QuantizedResNet50(nn.Module):
    """
    A class used to quantize resnet50

    Attributes
    ----------
    quant: torch.quantization.QuantStub
        convert the incoming float tensors to quantized tensors

    dequant: torch.quantization.DeQuantStub
        convert the quantized tensors back to float tensors

    model_fp32: torchvision.models
        floating point 32 fused model
    """
    
    def __init__(self, model_fp32):
        """
        Parameters
        ----------
        model_fp32: torchvision.models
            floating point 32 fused model
        """

        super(QuantizedResNet50, self).__init__()
        self.quant = QuantStub() # convert the incoming float tensors to quantized tensors.
        self.dequant = DeQuantStub() # convert the quantized tensors back to float tensors.
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def get_model(dataset, num_classes, path_to_saved_model):
    """
    Get the Pre-Trained ResNet-50 Model

    Parameters
    ----------
    num_classes: int
        number of classes in the dataset

    dataset: str
        the dataset chosen by the user 

    path_to_saved_model: str
        path to the saved model
        
    Return
    ------
    resnet_50: torchvision.models
        resnet_50 resnet-50 models
    """

    resnet_50 = resnet50(pretrained=True)
    if(dataset == 'emnist'):
        resnet_50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #resnet expects a 3-channel RGB image but EMNIST is a single channel grayscale image
    
    in_features = resnet_50.fc.in_features
    resnet_50.fc = nn.Linear(in_features, num_classes) # ResNet-50 is originally trained on imageNet's 1000 classes datasets tested have varying numbers of classes
    
    resnet_50.load_state_dict(torch.load(path_to_saved_model))
    return resnet_50

def fuse_model(fused, original):
    original.train()
    fused.eval()

    # Fuse the model in place manually
    # Fuses a list of modules into a single module

    fused = fuse_modules(fused, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused.named_children():
        if("layer" in module_name):
            for basic_block_name, basic_block in module.named_children():
                fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        fuse_modules(sub_block, [["0", "1"]], inplace=True)

    return fused

def get_hyperparameters(dataset):
    """
    Get the hyperparameters based on the dataset

    Parameters
    ----------
    dataset: str
        the dataset chosen by the user

    Return
    ------
    epochs: int
        number of epochs to train over

    learning_rate: float
        learning rate for optimizer

    weight_decay: float
        weight decay for optimizer
    """

    if(dataset == "cifar10"):
        epochs = 6
        learning_rate = 1e-4
        betas = (0.9, 0.99)
        weight_decay = 5e-4

    elif(dataset == "emnist"):
        epochs = 6
        learning_rate = 1e-3
        betas = (0.9, 0.99)
        weight_decay = 1e-4 

    elif(dataset == "svhn"):
        epochs = 4
        learning_rate = 1e-3
        betas = (0.9, 0.99)
        weight_decay = 1e-4

    return epochs, learning_rate, betas, weight_decay

def resnet_quantized(trainloader, testloader, dataset):
    """
    create a ResNet-50 model and conduct training and testing on the chosen dataset

    Parameters
    ----------
    trainloader: torch.utils.data.DataLoader
        the iterable training dataset

    testloader: torch.utils.data.DataLoader
        the iterable testing dataset

    dataset: str
        the dataset chosen by the user 
    """

    gpu = "V100"
    path_to_model_weights = f"ResNetQuantized/{gpu}s/resnet_quantized_weights_{dataset}.pt"
    image_size = (1,) + tuple(trainloader.dataset[0][0].shape)
    if(dataset == "svhn"):
        class_list = list(set(testloader.dataset.labels))
    else:
        class_list = testloader.dataset.classes
    num_classes = len(class_list)

    epochs, learning_rate, betas, weight_decay = get_hyperparameters(dataset)
    resnet50 = get_model(dataset, num_classes, f"ResNet/resnet_weights_{dataset}.pt")
    resnet50.to("cpu")
    fused_model = fuse_model(copy.deepcopy(resnet50), resnet50)
    
    # Create quantized model
    quantized_model = QuantizedResNet50(model_fp32=fused_model)
    quantized_model.qconfig = get_default_qconfig("fbgemm") # setting up the model for quantization-aware training with a configuration that is optimized for server-side deployment on x86 architectures
    prepare_qat(quantized_model, inplace=True)

    # Train
    training_loss = train(model=quantized_model, trainloader=trainloader, epochs=epochs, learning_rate=learning_rate, betas=betas, weight_decay=weight_decay, save_to_path=path_to_model_weights)
    plot_losses(loss=training_loss, epochs=epochs, dataset=dataset, path="ResNetQuantized/")

    # Inference
    quantized_model = QuantizedResNet50(model_fp32=fused_model)
    quantized_model.qconfig = get_default_qconfig("fbgemm") # setting up the model for quantization-aware training with a configuration that is optimized for server-side deployment on x86 architectures
    prepare_qat(quantized_model, inplace=True)

    confusion_matrix, accuracy = test(model=quantized_model, path_to_network=path_to_model_weights, testloader=testloader, num_classes=num_classes)
    print(f"Validation Accuracy for ResNet-Quantized-50: {accuracy}")
    display_confusion_matrix(conf=confusion_matrix, class_list=class_list, accuracy=accuracy, dataset=dataset, path="ResNetQuantized/")

    # Print the size of the model
    print_size_of_model(path_to_model=path_to_model_weights, label="ResNet-Quantized-50")

    # Print FLOPs and Number of Parameters
    num_operations(model=quantized_model.to("cpu"), input_size=image_size, label="ResNet-Quantized-50")

    # https://medium.com/@jan_marcel_kezmann/master-the-art-of-quantization-a-practical-guide-e74d7aad24f9