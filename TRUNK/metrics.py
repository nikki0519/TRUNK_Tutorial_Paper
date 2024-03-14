# ----------------------------------------------------
# Name: metrics.py
# Purpose: Script to measure the metrics of the TRUNK network
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
from thop import profile
import os
import torch
import argparse
import graphviz
import json
from datasets import GenerateDataset
from pathDecisions import map_leaf_name_to_category
from model_by_dataset import get_model
from grouper import AverageSoftmax, SigmoidMembership
from main import get_hyperparameters
import torch

## Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for metrics.py")
device = torch.device(device)

def parser():
    """
    Get command-line arguments

    Return
    ------
    args: argparse.Namespace
        user arguments 
    """
    parser = argparse.ArgumentParser(description="Metrics to measure TRUNK")
    parser.add_argument("--dataset", type=str, help="emnist, svhn, cifar10", default="emnist")
    parser.add_argument("--model_backbone", type=str, help="vgg or mobilenet", default="mobilenet")
    parser.add_argument("--ablation_study", nargs=3, type=float, help="the starting, ending grouping volatilites, and its increment step. We study the impact of this range of coefficients on the testing accuracy.")
    parser.add_argument("--visualize", action="store_true", help="Save visual of the tree")
    parser.add_argument("--untrained_asl", action="store_true", help="Get ASL of untrained root")
    args = parser.parse_args()
    return args

def print_size_of_model(path_to_category, path_to_weights, label):
    """
    Print the size of the model in MB

    Parameters
    ----------
    path_to_category: list
        the longest path down the tree to a specific category

    path_to_weights: str
        path to the directory containing all the model weights

    label: str
        name of the category and model for printing

    Return
    ------
    size: float
        the size of the model
    """
    size = 0
    for node_value in path_to_category:
        if(isinstance(node_value, str) and ("sg" in node_value or "root" in node_value)):
            path_to_model = os.path.join(path_to_weights, f"{node_value}.pt")
            size += os.path.getsize(path_to_model)

    print(label + str(size/1e6))

def longest_path_down_tree(root):
    """
    Get the longest path down the tree

    Parameters
    ----------
    root: TreeNode
        the root node of the tree

    Return
    ------
    longest_path: list
        list containing the node values down the longest path in the tree
    """

    if(not root):
        return []
    longest_path = []
    
    def dfs(node, current_path):
        nonlocal longest_path
        current_path.append(node.value)

        if(not node.children):
            if(len(current_path) > len(longest_path)):
                longest_path = current_path.copy()

        for child in node.children:
            dfs(child, current_path)
        current_path.pop() # remove the current node before backtracking
    
    dfs(root, [])
    return longest_path

def shortest_path_down_tree(root):
    """
    Get the shortest path down the tree

    Parameters
    ----------
    root: TreeNode
        the root node of the tree

    Return
    ------
    shortest_path: list
        list containing the node values down the shortest path in the tree
    """

    if(not root):
        return []
    shortest_path = None
    
    def dfs(node, current_path):
        nonlocal shortest_path
        current_path.append(node.value)

        if(not node.children):
            if(shortest_path is None or len(current_path) < len(shortest_path)):
                shortest_path = current_path.copy()

        for child in node.children:
            dfs(child, current_path)
        current_path.pop() # remove the current node before backtracking
    
    dfs(root, [])
    return shortest_path

def update_leaf_names(nodes_dict, inverse_category_encoding, tree_path):
    """
    Change the leaf nodes' name from sg{} to the category it is responsible for 

    Parameters
    ----------
    nodes_dict: dict
        dictionary of all the treenodes 

    inverse_category_encoding: dict
        dictionary mapping the categoryID to the category name

    tree_path: str
        the path to the saved tree pickle file

    Return
    ------
    nodes_dict: dict
        updated dictionary of all the treenodes
    """

    leaf_name_mapping = map_leaf_name_to_category(tree_path)
    for node_value, node in nodes_dict.items():
        if(node_value in leaf_name_mapping):
            node.value = inverse_category_encoding[int(leaf_name_mapping[node_value])]
    return nodes_dict

def visualize_tree(path_to_outputs, root, graph=None):
    """
    Create a png image visualizing the tree

    Parameters
    ----------
    path_to_outputs: str
        the path to save the png image of the tree in its appropriate folder

    root: TreeNode
        the root of the tree

    graph: graphviz.Digraph [Optional]
        the graph object used to draw the tree
    """

    if graph is None:
        graph = graphviz.Digraph()
    
    graph.node(str(root.value))
    for child in root.children:
        graph.edge(str(root.value), str(child.value))
        visualize_tree(path_to_outputs, child, graph)
    
    save_path = f"{path_to_outputs}/tree_visual"
    graph.render(save_path, format='png', cleanup=True)

def num_operations(path_to_category, dataset, dict_inputs, label):
    """
    Compute the number of floating point operations and trainable parameters are involved
    
    Parameters
    ----------
    path_to_category: list
        list containing the path to either the shortest or longest path

    dataset: GenerateDataset
        the dataset containing necessary attributes and functions

    dict_inputs: dict
        dictionary of inputs (image size and number of classes) for each model

    label: str
        label used for printing
    """

    total_flops, total_params = 0, 0
    for node_value in path_to_category:
        if(isinstance(node_value, str) and ("sg" in node_value or "root" in node_value)):
            input_size = dict_inputs[node_value][0]
            num_classes = dict_inputs[node_value][1]
            model = get_model(torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True), dataset.model_backbone, num_classes, input_size, node_value, node_value, debug_flag=False)

            random_input = torch.randn((1,) + tuple(input_size))
            random_input = random_input.to(device)
            flops, params = profile(model, inputs=(random_input, ))
            
            total_flops += flops
            total_params += params
    
    print(f"The {label} path in the tree which is {path_to_category[0]} -> {path_to_category[-1]} involves {total_flops} floating point operations and {total_params} trainable parameters")

def untrained_root_asl(dataset):
    """
    Get the Average Softmax Likelihood (ASL) of an untrained root node for the CiFAR tree

    Parameters
    ----------
    dataset: GenerateDataset
        the dataset containing necessary attributes and functions
    """

    image_shape = dataset.image_shape
    num_classes = len(dataset.labels)
    model_backbone = dataset.model_backbone
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)

    model = get_model(dataloader, model_backbone, num_classes, image_shape, current_supergroup="root", supergroup="root", debug_flag=False)
    path_to_softmax_matrix = os.path.join(dataset.path_to_outputs, f"model_softmax/untrained_root_avg_softmax.pt")
    AverageSoftmax([model], dataloader, current_supergroup="root", softmax_file_path=path_to_softmax_matrix)

def display_asl_matrix(dataset):
    """
    Display the untrained and trained root ASL for CiFAR10 dataset

    Parameters
    ----------
    dataset: GenerateDataset
        the dataset containing necessary attributes and functions
    """

    path_to_untrained_asl = os.path.join(dataset.path_to_outputs, f"model_softmax/untrained_root_avg_softmax.pt")
    untrained_asl = torch.load(path_to_untrained_asl, map_location=torch.device("cpu"))

    path_to_trained_asl = os.path.join(dataset.path_to_outputs, f"model_softmax/root_avg_softmax.pt")
    trained_asl = torch.load(path_to_trained_asl, map_location=torch.device("cpu"))

    category_encoding = dataset.get_integer_encoding()
    dog_idx, plane_idx = category_encoding["dog"], category_encoding["airplane"]

    untrained_dog_softmax = untrained_asl[dog_idx, :]
    trained_dog_softmax = trained_asl[dog_idx, :]

    untrained_plane_softmax = untrained_asl[plane_idx, :]
    trained_plane_softmax = trained_asl[plane_idx, :]

    assert untrained_dog_softmax.shape == trained_dog_softmax.shape, f"dog: untrained shape {untrained_dog_softmax.shape} != trained shape {trained_dog_softmax.shape}"
    assert untrained_plane_softmax.shape == trained_plane_softmax.shape, f"plane: untrained shape {untrained_plane_softmax.shape} != trained shape {trained_plane_softmax.shape}"

    final_tensor = torch.stack([untrained_dog_softmax, untrained_plane_softmax, trained_dog_softmax, trained_plane_softmax], dim=0)
    print(category_encoding)
    print("=============")
    print(final_tensor)

def sigmoid_membership(dataset):
    """
    Display the sigmoid membership of the trained root node for the dog category in the CIFAR dataset 

    Parameters
    ----------
    dataset: GenerateDataset
        the dataset containing necessary attributes and functions
    """

    path_to_trained_asl = os.path.join(dataset.path_to_outputs, f"model_softmax/root_avg_softmax.pt")
    trained_asl = torch.load(path_to_trained_asl, map_location=torch.device("cpu"))

    category_encoding = dataset.get_integer_encoding()
    dog_idx = category_encoding["dog"]
    trained_dog_softmax = trained_asl[dog_idx, :]

    hyperparameters = json.load(open(os.path.join(dataset.path_to_outputs, "hyperparameters.json"), "r"))
    grouping_volatility = hyperparameters["grouping_volatility"]
    num_classes = len(dataset.labels)

    membership = []
    for asl in trained_dog_softmax:
        membership.append(SigmoidMembership(num_classes, grouping_volatility, float(asl)))

    membership = {list(category_encoding.keys())[idx]: membership[idx] for idx in range(len(membership))}
    print(membership)

if __name__ == "__main__":
    args = parser()
    config = get_hyperparameters(f"./Datasets/{args.dataset.lower()}/{args.model_backbone.lower()}")

    if(args.ablation_study):
        list_of_grouping_volatilities = [idx/100 for idx in range(int(args.ablation_study[0]*100), int(args.ablation_study[1]*100), int(args.ablation_study[2]*100))]
    else:
        list_of_grouping_volatilities = [config.general.grouping_hyperparameters.grouping_volatility]

    for grouping_volatility in list_of_grouping_volatilities:
        print(f"------Current Grouping Volatility: {grouping_volatility}---------")
        dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), config=config, grouping_volatility=grouping_volatility, train=False)

        tree_path = os.path.join(dataset.path_to_outputs, "tree.pkl") # Path to the tree based on the particular dataset and model backbone used
        inverse_category_encoding = dataset.get_inverse_integer_encoding()
        nodes_dict = update_leaf_names(dataset.get_dictionary_of_nodes(), inverse_category_encoding, tree_path)

        if(args.visualize):
            visualize_tree(dataset.path_to_outputs, nodes_dict["root"])
    
        longest_path = longest_path_down_tree(nodes_dict["root"])
        shortest_path = shortest_path_down_tree(nodes_dict["root"])

        path_to_model_weights = os.path.join(dataset.path_to_outputs, f"model_weights")
        print_size_of_model(shortest_path, path_to_model_weights, label=f"Shortest path (root -> {shortest_path[-1]}) size (MB): ")
        print_size_of_model(longest_path, path_to_model_weights, label=f"Longest path (root -> {longest_path[-1]}) size (MB): ")

        dict_model_inputs = json.load(open(os.path.join(dataset.path_to_outputs, "model_weights/inputs_to_models.json")))
        num_operations(shortest_path, dataset, dict_model_inputs, label="shortest")
        num_operations(longest_path, dataset, dict_model_inputs, label="longest")

        if(args.untrained_asl):
            untrained_root_asl(dataset)
            display_asl_matrix(dataset)
            sigmoid_membership(dataset)
        