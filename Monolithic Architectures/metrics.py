# ----------------------------------------------------
# Name: metrics.py
# Purpose: Script to measure the metrics of the TRUNK network
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import os
import torch
import time
from functools import wraps
from thop import profile
import matplotlib.pyplot as plt
import seaborn as sns

def print_size_of_model(path_to_model, label):
    """
    Print the size of the model in MB

    Parameters
    ----------
    path_to_model: str
        the path to the saved model

    label: str
        name of the model for printing

    Return
    ------
    size: float
        the size of the model
    """

    size = os.path.getsize(path_to_model)
    print("model: ",label,' \t','Size (MB):', size/1e6)

def run_time(function):
    """
    Wrapper function to compute run-time for any function

    Parameters
    ----------
    function: any function

    Return
    ------
    run_time_wrapper: wraps
        wrapper function
    """

    @wraps(function)
    def run_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {function.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result
    return run_time_wrapper

def num_operations(model, input_size, label):
    """
    Measure the total number of floating point operations (FLOPs) performed by the model and the number of trainable parameters in the model

    Parameters
    ----------
    model: Custom Model
        the model we are measuring

    input_size: tuple
        dummy input size

    label: str
        name of the model for printing
    """

    input_size = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_size, ))
    print(f"For model {label}, FLOPs: {flops}, Params: {params}")

def plot_losses(loss, epochs, dataset, path):
    """
    Plot the training loss

    Parameters
    ----------
    loss: list
        list of training loss over all the epochs and batches

    epochs: int
        total number of epochs

    dataset: str
        the dataset chosen by the user 

    path: str
        path to store losses
    """

    plt.plot(range(len(loss)), loss, label="Training Loss")

    plt.title(f"Loss per Iteration")
    plt.xlabel(f"Iterations over {epochs} epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    plt.savefig(f'{path}/training_loss_{dataset}.png')

def display_confusion_matrix(conf, class_list, accuracy, dataset, path):
    """
    conf: np.array
        the confusion matrix for the dataset using this trained model

    class_list: list
        list of categories

    accuracy: float
        the accuracy of the model

    dataset: str
        the dataset chosen by the user 

    path: str
        path to store conf matrix
    """
    plt.clf()
    sns.heatmap(conf, xticklabels=class_list, yticklabels=class_list, annot=True)
    plt.xlabel(f"True Label \n Accuracy: {accuracy}")
    plt.ylabel("Predicted Label")

    plt.savefig(f'{path}/confusion_matrix_{dataset}.png')