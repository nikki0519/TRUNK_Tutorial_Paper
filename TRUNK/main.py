# ----------------------------------------------------
# Name: main.py
# Purpose: Script to execute the necessary function calls based on the user's needs
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary packages
import os
import argparse
from datasets import GenerateDataset, get_dataloader
from model_by_dataset import get_model
from train import train
from test import test, display_confusion_matrix, ablation_study
from grouper import AverageSoftmax, update_target_map
from collections import deque
import torch
import json
import time
from omegaconf import OmegaConf
import numpy as np

# Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for main.py")
device = torch.device(device)

def parser():
    """
    Get command-line arguments

    Return
    ------
    args: argparse.Namespace
        user arguments 
    """
    
    parser = argparse.ArgumentParser(description="TRUNK for Image Classification")
    parser.add_argument("--train", action="store_true", help="Conduct training")
    parser.add_argument("--infer", action="store_true", help="Conduct inference")
    parser.add_argument("--dataset", type=str, help="emnist, svhn, cifar10", choices=["emnist", "svhn", "cifar10"], default="emnist")
    parser.add_argument("--model_backbone", type=str, help="vgg or mobilenet", choices=["vgg", "mobilenet"], default="mobilenet")
    parser.add_argument("--grouping_volatility", action="store_true", help="If this is set true, then we will use the grouping volatility indicated in the config file")
    parser.add_argument("--ablation_study", nargs=3, type=float, help="the starting, ending grouping volatilites, and its increment step. We study the impact of this range of coefficients on the testing accuracy.")
    parser.add_argument("--debug", action="store_true", help="Print information for debugging purposes")
    args = parser.parse_args()
    return args

def get_hyperparameters(path):
    """
    Get all the hyperparameters required for the dataset we want to train

    Parameters
    ----------
    path: str
        relative path to config file

    Return
    ------
    hyperparameters: dict
        dictionary of all the hyperparameters
    """

    path_to_config = os.path.join(path, "hyperparameters.yaml")
    config = OmegaConf.load(path_to_config)
    return config

def get_list_of_models_by_path(dataloader, model_backbone, current_supergroup, dictionary_of_inputs_for_models, debug_flag=False):
    """
    Get a list of models that are needed for the current supergroup

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        iterable dataloader for the custom dataset

    current_supergroup: str
        current supergroup we're at in the tree

    model_backbone: str
        based on user input of whether we want to use vgg or MobileNet

    dictionary_of_inputs_for_models: dict
        dictionary mapping the supergroup and the number of classes and image shape they require (key, value) = (supergroup name, (image_shape, num_classes))

    debug_flag: bool [Optional]
        print outputs if debug=True

    Return
    ------
    list_of_models: list
        return a list of all the models involved in the path to the current supergroup
    """

    paths = dataloader.dataset.get_paths()
    list_of_groups = paths[current_supergroup]
    
    list_of_models = []
    for idx, supergroup in enumerate(list_of_groups):
        image_shape, num_classes = dictionary_of_inputs_for_models[supergroup]
        model = get_model(dataloader=dataloader, model_backbone=model_backbone.lower(), number_of_classes=num_classes, image_shape=image_shape, current_supergroup=current_supergroup, supergroup=supergroup, debug_flag=debug_flag)
        list_of_models.append(model)
    return list_of_models

def skip_current_node(dataloader, current_supergroup):
    """
    Check if the current node's parent has been trained or not. If it is not trained then we will enqueue this node again and check the next one

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        iterable dataloader of the dataset we are using

    current_supergroup: str
        the current_supergroup we are examining

    Return
    ------
    skip: bool
        check if the current node's parent has been trained or not
    """

    nodes_dict = dataloader.dataset.get_dictionary_of_nodes() # get the dictionary of all the nodes
    current_node = nodes_dict[current_supergroup] # the current TreeNode based on the value

    skip = False
    if(current_supergroup == "root"):
        skip = False
    elif(current_node.num_groups <= 1):
        skip = True
    elif(current_node.parent and current_node.parent.is_trained == False):
        skip = True

    return skip

def check_num_classes(nodes_dict, supergroup_queue):
    """
    If all the nodes in the queue only have one class then they are all leaf nodes and we are done training the tree

    Parameters
    ----------
    supergroup_queue: queue
        the queue of nodes left to train

    nodes_dict: dict
        dictionary of nodes in the tree

    Return
    ------
    skip: bool
        return True if all the nodes are a leaf nodes
    """
    
    for node_value in supergroup_queue:
        node = nodes_dict[node_value]
        if(node.num_groups > 1):
            print(f"{node_value} is still a supergroup with number of classes it is responsible for are {len(node.classes)}")
            return False
    return True

def update_inputs_for_model(nodes_dict, image_shape):
    """
    Since the tree keeps changing, we need to verify the number of classes each node is responsible for 

    Parameters
    ----------
    nodes_dict: dict
        dictionary of nodes in the tree

    image_shape: tuple
        the input image shape to the model, only used when the current supergroup is the root

    Return
    ------
    dictionary_of_inputs_for_models: dict
        updated dictionary of inputs to the model
    """

    dictionary_of_inputs_for_models = {}
    for node_value, node in nodes_dict.items():
        if(node_value == "root"):
            if(node.is_trained):
                dictionary_of_inputs_for_models[node_value] = [image_shape, node.num_groups]
            else:
                dictionary_of_inputs_for_models[node_value] = [image_shape, len(node.classes)]
        else:
            if(node.is_trained):
                dictionary_of_inputs_for_models[node_value] = [node.parent.output_image_shape, node.num_groups]
            else:
                dictionary_of_inputs_for_models[node_value] = [node.parent.output_image_shape, len(node.classes)]
    
    return dictionary_of_inputs_for_models

def format_time(runtime):
    """
    Output time in the format hh:mm:ss

    Parameters
    ----------
    runtime: float
        total runtime of script

    Return
    ------
    formatted_time: str
        time in the appropriate format
    """

    # Convert runtime into hours, minutes, and seconds
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"

def main():
    start_time = time.time()
    args = parser()
    config = get_hyperparameters(f"./Datasets/{args.dataset.lower()}/{args.model_backbone.lower()}")

    ## set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(config.seed)

    if(args.train):
        ### Training the entire tree
        # Iterate through grouping volatilities for an ablation study
        if(args.grouping_volatility):
            list_of_grouping_volatilities = [config.general.grouping_hyperparameters['grouping_volatility']]
        else:
            list_of_grouping_volatilities = [idx/100 for idx in range(int(args.ablation_study[0]*100), int(args.ablation_study[1]*100), int(args.ablation_study[2]*100))]

        for grouping_idx, grouping_volatility in enumerate(list_of_grouping_volatilities):
            print(f"Current Grouping Volatility is {grouping_volatility}")

            # Download datasets
            train_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), config, grouping_volatility, train=True, validation=False)
            test_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), config, grouping_volatility, train=True, validation=True)

            # Dataset features
            class_labels = train_dataset.labels # List of unique classes in the dataset
            image_shape = train_dataset.image_shape # Get the shape of the image in the dataset (1xCxHxW)

            # Create dataloaders
            trainloader = get_dataloader(train_dataset, config, train=True, validation=False)
            testloader = get_dataloader(test_dataset, config, train=True, validation=True)
            supergroup_queue = deque(["root"]) # queue to keep track of all the supergroups to train

            while(supergroup_queue):
                paths = trainloader.dataset.get_paths()
                nodes_dict = trainloader.dataset.get_dictionary_of_nodes()

                if(args.debug):
                    print(f"Number of Nodes Left: {len(supergroup_queue)}")

                if(check_num_classes(nodes_dict, supergroup_queue)):
                    # If all the supergroups in the queue have one class then theyre all leaf nodes
                    if(args.debug):
                        print(f"All the remaining nodes in the queue, {supergroup_queue} are leaf nodes")
                    break

                current_supergroup = supergroup_queue.popleft()
                dictionary_of_inputs_for_models = update_inputs_for_model(nodes_dict, trainloader.dataset.image_shape)
                if(skip_current_node(trainloader, current_supergroup)): 
                    # check if the parent node of this current node has been trained. If not then skip
                    if(args.debug):
                        print(f"current supergroup: {current_supergroup} added back in the queue with num children: {dictionary_of_inputs_for_models[current_supergroup][1]}")
                    supergroup_queue.append(current_supergroup)
                    continue

                print(f"Currently on supergroup {current_supergroup} which needs an image shape of {dictionary_of_inputs_for_models[current_supergroup][0]} and has {dictionary_of_inputs_for_models[current_supergroup][1]} classes")
                print(f"The path to {current_supergroup} is {paths[current_supergroup]}")

                # Get all the models and the model weights for the current supergroup of the MNN tree
                list_of_models = get_list_of_models_by_path(dataloader=trainloader, model_backbone=args.model_backbone, current_supergroup=current_supergroup, dictionary_of_inputs_for_models=dictionary_of_inputs_for_models, debug_flag=args.debug)

                # Train the current supergroup of the MNN tree
                ## Create the models_weights directory if it doesn't exist
                if(not os.path.exists(os.path.join(trainloader.dataset.path_to_outputs, f"model_weights"))):
                    os.makedirs(os.path.join(trainloader.dataset.path_to_outputs, f"model_weights"))
                path_save_model = os.path.join(trainloader.dataset.path_to_outputs, f"model_weights/{current_supergroup}.pt")
                
                print(f"Training Started on Module {current_supergroup}")
                if(current_supergroup in config):
                    class_hyperparameters = config[current_supergroup].class_hyperparameters
                else:
                    class_hyperparameters = config.general.class_hyperparameters

                image_shape = train(list_of_models=list_of_models, current_supergroup=current_supergroup, config=class_hyperparameters, grouping_volatility=grouping_volatility, model_save_path=path_save_model, trainloader=trainloader, validationloader=testloader)
                image_shape = tuple(image_shape[1:]) # change from (BxCxHxW) -> (CxHxW)

                # Create the average softmax of this current trained supergroup
                print("Computing Average SoftMax")
                checkpoint = torch.load(path_save_model)
                list_of_models[-1].load_state_dict(checkpoint['model_state_dict'])

                path_to_softmax_matrix = os.path.join(trainloader.dataset.path_to_outputs, f"model_softmax/{current_supergroup}_avg_softmax.pt")
                ## Create the model_softmax directory if it doesn't exist
                if(not os.path.exists(os.path.join(trainloader.dataset.path_to_outputs, f"model_softmax"))):
                    os.makedirs(os.path.join(trainloader.dataset.path_to_outputs, f"model_softmax"))
                AverageSoftmax(list_of_models, trainloader, current_supergroup, path_to_softmax_matrix)

                # Update the target_map based on the softmax of the current supergroup
                print("Updating TargetMap")
                ## Get supergroup training hyperparameters 
                if(current_supergroup in config):
                    grouping_hyperparameters = config[current_supergroup].grouping_hyperparameters
                else:
                    grouping_hyperparameters = config.general.grouping_hyperparameters

                if(args.grouping_volatility):
                    supergroup_grouping_volatility = grouping_hyperparameters.grouping_volatility
                else:
                    supergroup_grouping_volatility = grouping_volatility

                print(f"For the current supergroup: {current_supergroup}, the grouping volatility we are using is: {supergroup_grouping_volatility}")
                path_decisions = trainloader.dataset.get_path_decisions() # the paths down the tree from the root node to each node in the tree
                list_of_new_supergroups = update_target_map(trainloader, current_supergroup, supergroup_grouping_volatility, path_to_softmax_matrix, path_decisions[current_supergroup], debug=args.debug)
                nodes_dict = trainloader.dataset.get_dictionary_of_nodes() # updated dictionary of nodes

                # If there are new supergroups then add it to the end of the queue and update the dictionary_of_inputs_for_models
                if(list_of_new_supergroups):
                    supergroup_queue.extend(list_of_new_supergroups)
                    if(args.debug):
                        print(f"List of Supergroups to still train: {supergroup_queue}")
                supergroup_queue = deque([sg for sg in supergroup_queue if sg in list(nodes_dict.keys())]) # update the queue to remove supergroups that have been grouped with other supergroups

                # Re-train the current supergroup to distinguish betweeen the number of children it has rather than the number of classes 
                node = nodes_dict[current_supergroup]
                num_children = node.num_groups 
                num_classes = len(node.classes) 
        
                print(f"For the current supergroup: {current_supergroup}, they have {num_classes} labels and {num_children} children with an image shape of {dictionary_of_inputs_for_models[current_supergroup][0]}")
                if(num_children != num_classes):
                    print(f"Re-training on the current supergroup {current_supergroup} with number of children: {num_children}")
                    dictionary_of_inputs_for_models[current_supergroup][1] = num_children # changing the number of groups for the model to distinguish between
                    os.remove(path_save_model) # Automatically remove the previously stored weights for this node
                    list_of_models = get_list_of_models_by_path(dataloader=trainloader, model_backbone=args.model_backbone, current_supergroup=current_supergroup, dictionary_of_inputs_for_models=dictionary_of_inputs_for_models, debug_flag=args.debug) # reset the weights of the current supergroup

                    image_shape = train(list_of_models=list_of_models, current_supergroup=current_supergroup, config=grouping_hyperparameters, grouping_volatility=grouping_volatility, model_save_path=path_save_model, trainloader=trainloader, validationloader=testloader)
                    image_shape = tuple(image_shape[1:]) # change from (BxCxHxW) -> (CxHxW)

                nodes_dict[current_supergroup].output_image_shape = image_shape
                nodes_dict[current_supergroup].is_trained = True
                trainloader.dataset.update_tree_attributes(nodes_dict) # update these attributes in the tree

            # Record the inputs to each supergroup model to a json file
            with open(os.path.join(trainloader.dataset.path_to_outputs, "model_weights/inputs_to_models.json"), "w") as fptr:
                fptr.write(json.dumps(dictionary_of_inputs_for_models, indent=4))

        end_time = time.time()
        print(f"Finished Training TRUNK in " + format_time(end_time - start_time))

    if(args.infer):
        ### Conduct inference on the trained tree
        # Download datasets and create dataloader
        list_of_accuracies = []

        # Iterate through grouping volatilities for an ablation study
        if(args.grouping_volatility):
            list_of_grouping_volatilities = [config.general.grouping_hyperparameters['grouping_volatility']]
        else:
            list_of_grouping_volatilities = [idx/100 for idx in range(int(args.ablation_study[0]*100), int(args.ablation_study[1]*100), int(args.ablation_study[2]*100))]

        for grouping_idx, grouping_volatility in enumerate(list_of_grouping_volatilities):
            print(f"Current Grouping Volatility is {grouping_volatility}")

            # Download datasets
            test_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), config, grouping_volatility, train=False, validation=False)
            # Create dataloaders
            testloader = get_dataloader(test_dataset, config)
            
            # conduct inference on TRUNK using the testing dataset
            confusion_matrix, accuracy = test(testloader)
            list_of_accuracies.append(accuracy)
            display_confusion_matrix(confusion_matrix, testloader)

        if(args.ablation_study):
            # Save accuracies to a file
            path = f"./Datasets/{args.dataset.lower()}/{args.model_backbone.lower()}/accuracies.txt"
            with open(path, "w") as fptr:
                for acc in list_of_accuracies:
                    fptr.write(str(acc) + "\n")

            # visualize the ablation study of grouping volatilities and inference accuracies
            ablation_study(list_of_grouping_volatilities, list_of_accuracies, testloader)
        else:
            print(f"Testing accuracy is: {list_of_accuracies[0]} for dataset {args.dataset}")

        end_time = time.time()
        print(f"Finished Testing TRUNK in " + format_time(end_time - start_time))

if __name__ == "__main__":
    main()