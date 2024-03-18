# ----------------------------------------------------
# Name: test.py
# Purpose: Script to conduct inference on the chosen dataset using TRUNK
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from Datasets.emnist.models.MobileNet_Backbone import MNN as emnist_mobilenet
from Datasets.emnist.models.VGGNet_Backbone import MNN as emnist_vgg
from Datasets.cifar10.models.MobileNet_Backbone import MNN as cifar10_mobilenet
from Datasets.cifar10.models.VGGNet_Backbone import MNN as cifar10_vgg
from Datasets.svhn.models.MobileNet_Backbone import MNN as svhn_mobilenet
from Datasets.svhn.models.VGGNet_Backbone import MNN as svhn_vgg

## Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for test.py")
device = torch.device(device)

def get_model(dataloader, current_supergroup):
    """
    Get the current supergroup model

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        iterable dataloader

    current_supergroup: str
        the current supergroup we're at

    Return
    ------
    model: TRUNK model depending on dataset and backbone
    """

    def read_json_file():
        with open(path_to_model_inputs, "r") as fptr:
            return json.load(fptr)

    path_to_model_weights = os.path.join(dataloader.dataset.path_to_outputs, f"model_weights")
    path_to_model_inputs = os.path.join(path_to_model_weights, "inputs_to_models.json")
    path_to_current_sg_weights = os.path.join(path_to_model_weights, f"{current_supergroup}.pt")
    
    inputs_to_models = read_json_file()
    image_shape, number_of_classes = inputs_to_models[current_supergroup]

    model_backbone = dataloader.dataset.model_backbone
    dataset = dataloader.dataset.dataset 

    if(dataset.lower() == "emnist"):
        if(model_backbone.lower() == "mobilenet"):
            model = emnist_mobilenet(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
        
        elif(model_backbone.lower() == "vgg"):
            model = emnist_vgg(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
    
    elif(dataset.lower() == "cifar10"):
        if(model_backbone.lower() == "mobilenet"):
            model = cifar10_mobilenet(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
        
        elif(model_backbone.lower() == "vgg"):
            model = cifar10_vgg(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
    
    elif(dataset.lower() == "svhn"):
        if(model_backbone.lower() == "mobilenet"):
            model = svhn_mobilenet(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
        
        elif(model_backbone.lower() == "vgg"):
            model = svhn_vgg(supergroup=current_supergroup, number_of_classes=number_of_classes, input_shape=image_shape, debug_flag=False)
    
    else:
        raise Exception("Please provide a valid dataset, i.e. emnist, cifar10, or svhn")

    checkpoint = torch.load(path_to_current_sg_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # model.load_state_dict(torch.load(path_to_current_sg_weights)) # uncomment for batchNorm folder
    # model = model.to(device)
    return model

def test(testloader):
    """
    Conduct inference on the trained TRUNK model

    Parameters
    ----------
    testloader: torch.utils.data.DataLoader
        iterable test dataloader

    Return
    ------
    confusion_matrix: np.array
        the confusion matrix for the chosen dataset on the trained model

    accuracy: float
        Accuracy evaluated using the testing dataset
    """
    num_classes = len(testloader.dataset.labels)
    confusion_matrix = np.zeros((num_classes, num_classes))

    inverse_target_map = testloader.dataset.get_inverse_target_map() 
    inverse_path_decisions = testloader.dataset.get_inverse_path_decisions() 
    leaf_nodes = testloader.dataset.get_leaf_nodes() 

    num_right = 0 
    total = 0 

    with tqdm(enumerate(testloader), total=len(testloader), desc="Batch Idx:") as progress_bar:
        for batch_idx, (image, target_map) in progress_bar:
            image = image.to(device)
            depth = 0 # The depth or layer of the tree we're currently at for a category
            current_node = target_map[depth].to(device) 
            path_taken = [] # path taken by the model for the current batch of images
            model = get_model(testloader, current_supergroup="root") 

            while(True):
                model.eval()
                image, sg_prediction = model.evaluate(image)
                next_sg = torch.argmax(sg_prediction, dim=1).item()
                path_taken.append(next_sg)

                if(path_taken in list(leaf_nodes.values()) or inverse_path_decisions == {(): "root"}): # check the second part on 0.6-0.63
                    # the path taken has led us down to a leaf node in the tree
                    target_map_integer = [x.item() for x in target_map if x.item() != -1] 
                    predicted_class = inverse_target_map[tuple(path_taken)]
                    target_class = inverse_target_map[tuple(target_map_integer)]
                    confusion_matrix[int(predicted_class)][int(target_class)] += 1

                    num_right += next_sg == current_node.squeeze().item()
                    total += 1
                    break
                
                if(next_sg != current_node):
                    # If this condition is true, the prediction is incorrect
                    total += 1
                    break
                
                del model # delete the previous model and load a new one
                model = get_model(testloader, current_supergroup=inverse_path_decisions[tuple(path_taken)])
                depth += 1
                current_node = target_map[depth].to(device)

            progress_bar.set_description(f"Batch Idx: {batch_idx}/{len(testloader)}")

        accuracy = num_right / (total + 1e-5) * 100.0 # we add 1e-5 to denominator to avoid dividing by 0
    return confusion_matrix, accuracy

def display_confusion_matrix(confusion_matrix, testloader):
    """
    Save the confusion matrix

    Parameters
    ----------
    confusion_matrix: np.array
        the confusion matrix for the dataset using this trained model

    testloader: torch.utils.data.DataLoader
        iterable test dataset
    """
    class_list = testloader.dataset.labels
    path = testloader.dataset.path_to_outputs

    sum_of_columns = confusion_matrix.sum(axis=0)
    confusion_matrix /= sum_of_columns

    plt.figure()
    sns.heatmap(confusion_matrix, xticklabels=class_list, yticklabels=class_list, annot=True)
    plt.xlabel(f"True Label")
    plt.ylabel("Predicted Label")

    plt.savefig(f'{path}/confusion_matrix.png')

def ablation_study(grouping_volatilities, accuracies, testloader):
    """
    Graph the inference accuracy obtained for each grouping volatility if using the CIFAR-10 dataset

    Parameters
    ----------
    grouping_volatilities: list
        list of grouping volatilities

    accuracies: list
        list of inference accuracies obtained on the testing dataset

    testloader: torch.utils.data.DataLoader
        iterable test dataset
    """

    plt.figure()
    plt.plot(grouping_volatilities, accuracies, marker='o', color='red')
    plt.title('Analyzing the effects of the grouping volatility which structures the tree on the testing accuracy')
    plt.xlabel('Grouping Volatility')
    plt.ylabel('Inference Accuracy [%]')

    path = f"./Datasets/{testloader.dataset.dataset}/{testloader.dataset.model_backbone}"
    plt.savefig(f'{path}/ablation_study.png')