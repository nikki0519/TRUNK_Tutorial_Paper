# ----------------------------------------------------
# Name: model_by_dataset.py
# Purpose: Script to gather the appropriate network based on the dataset and supergroup
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
from Datasets.emnist.models.MobileNet_Backbone import MNN as emnist_mobilenet
from Datasets.emnist.models.VGGNet_Backbone import MNN as emnist_vgg
from Datasets.cifar10.models.MobileNet_Backbone import MNN as cifar10_mobilenet
from Datasets.cifar10.models.VGGNet_Backbone import MNN as cifar10_vgg
from Datasets.svhn.models.MobileNet_Backbone import MNN as svhn_mobilenet
from Datasets.svhn.models.VGGNet_Backbone import MNN as svhn_vgg
import os
import torch

## Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for model_by_dataset.py")
device = torch.device(device)

def get_model(dataloader, model_backbone, number_of_classes, image_shape, current_supergroup, supergroup="root", debug_flag=True):
    """
    Get the model based on the backbone chosen by the user and the supergroup we're looking at

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        Iterable dataloader used to train

    model_backbone: str
        model backbone chosen by the user (i.e. vgg or MobileNet)

    number_of_classes: int
        number of classes to differentiate between at the current supergroup

    image_shape: tuple
        shape of the image (1xCxHxW)

    current_supergroup: str
        the current supergroup we're at in the tree

    supergroup: str
        the supergroup we're at on our way to the current super_group

    debug_flag: bool
        print outputs if debug=True

    Return
    ------
    model: emnist_mobilenet or emnist_vgg or svhn_mobilent or svhn_emnist
        the model with its weights
    """

    supergroup_weights = os.path.join(dataloader.dataset.path_to_outputs, f"model_weights/{supergroup}.pt")

    if(model_backbone == "mobilenet"):
        if(dataloader.dataset.dataset == "emnist"):
            model = emnist_mobilenet(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        elif(dataloader.dataset.dataset == "cifar10"):
            model = cifar10_mobilenet(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        elif(dataloader.dataset.dataset == "svhn"):
            model = svhn_mobilenet(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        else:
            raise Exception("Provide correct dataset like emnist or cifar10 or svhn")
    elif(model_backbone == "vgg"):
        if(dataloader.dataset.dataset == "emnist"):
            model = emnist_vgg(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        elif(dataloader.dataset.dataset == "cifar10"):
            model = cifar10_vgg(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        elif(dataloader.dataset.dataset == "svhn"):
            model = svhn_vgg(supergroup=supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=debug_flag)
        else:
            raise Exception("Provide correct dataset like emnist or cifar10 or svhn")
    else:
        raise Exception("Provide a model backbone")
    
    model = model.to(device)
    if(current_supergroup != supergroup):
        checkpoint = torch.load(supergroup_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model