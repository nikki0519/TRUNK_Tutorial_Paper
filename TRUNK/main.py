# Import necessary packages
import os
import argparse
from datasets import GenerateDataset, get_dataloader
from model_by_dataset import get_model
from train import train
from test import test, display_confusion_matrix
from grouper import AverageSoftmax, update_target_map
from collections import deque
import torch
import json
import time
from lightning.fabric import Fabric

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
    parser.add_argument("--improve_model_weights", type=str, help="Improve a certain supergroup's validation accuracy by optimizing its parameters")
    parser.add_argument("--train_batch_size", type=int, help="Training Batch Size", default=128)
    parser.add_argument("--eval_batch_size", type=int, help="Evaluation Batch Size", default=128)
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers for dataloader", default=0)
    parser.add_argument("--dataset", type=str, help="emnist, svhn, cifar10", default="emnist")
    parser.add_argument("--model_backbone", type=str, help="vgg or mobilenet", default="mobilenet")
    parser.add_argument("--debug", action="store_true", help="Print information for debugging purposes")
    args = parser.parse_args()
    return args

def get_hyperparameters(dataloader):
    """
    Get all the hyperparameters required for the dataset we want to train

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        Iterable dataloader

    Return
    ------
    hyperparameters: dict
        dictionary of all the hyperparameters
    """

    path_to_hyperparameters = os.path.join(dataloader.dataset.path_to_outputs, "hyperparameters.json")
    fptr = open(path_to_hyperparameters, "r")
    return json.load(fptr)

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

    paths = dataloader.dataset.get_paths() # get a dictionary mapping the paths to any supergroup from the root node
    list_of_groups = paths[current_supergroup] # get only the modules/nodes on the path to the current supergroup
    
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

def improve_modules_accuracy(trainloader, testloader, current_supergroup):
    """
    Improve the accuracy of the module based on the updated hyper-parameters provided in the json file 

    Parameters
    ----------
    trainloader: torch.utils.data.DataLoader
        iterable training dataset

    testloader: torch.utils.data.DataLoader
        iterable testing dataset

    current_supergroup: TreeNode
        Re-training the current supergroup to improve the validation accuracy based on the new hyperparameters
    """
    if(current_supergroup is None):
        return

    nodes_dict = trainloader.dataset.get_dictionary_of_nodes()
    inputs_to_model = update_inputs_for_model(nodes_dict, trainloader.dataset.image_shape)
    hyperparameters = get_hyperparameters(trainloader)
    list_of_models = get_list_of_models_by_path(trainloader, trainloader.dataset.model_backbone, current_supergroup.value, inputs_to_model)

    optimizer = torch.optim.Adam(list_of_models[-1].parameters(), lr=hyperparameters[f'{current_supergroup.value}_learning_rate'], weight_decay=hyperparameters[f'{current_supergroup.value}_weight_decay'])
    path_save_model = os.path.join(trainloader.dataset.path_to_outputs, f"model_weights/{current_supergroup.value}.pt")

    if(os.path.exists(path_save_model)):
        image_shape = train(list_of_models=list_of_models, current_supergroup=current_supergroup.value, epochs=hyperparameters[f'{current_supergroup.value}_epochs'], optimizer=optimizer, model_save_path=path_save_model, trainloader=trainloader, validationloader=testloader)
        current_supergroup.output_image_shape = image_shape
        trainloader.dataset.update_tree_attributes(nodes_dict) # update these attributes in the tree
    else:
        raise Exception("Model weights or hyper-parameters do not exist so we can't re-train")

    for child in current_supergroup.children:
        if(child.value not in trainloader.dataset.get_leaf_nodes()):
            improve_modules_accuracy(trainloader, testloader, child)


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
    if(args.improve_model_weights):
        ### Improving the validation accuracy of a trained module
        # Download datasets
        train_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), train=True, re_train=True)
        test_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), train=False, re_train=True)

        # Create dataloaders
        trainloader = get_dataloader(train_dataset, args.train_batch_size, args.num_workers, shuffle=True)
        testloader = get_dataloader(test_dataset, args.eval_batch_size, args.num_workers, shuffle=True)
        
        nodes_dict = trainloader.dataset.get_dictionary_of_nodes()
        improve_modules_accuracy(trainloader, testloader, current_supergroup=nodes_dict[args.improve_model_weights])

        nodes_dict = trainloader.dataset.get_dictionary_of_nodes() # updated nodes_dict
        inputs_to_model = update_inputs_for_model(nodes_dict, trainloader.dataset.image_shape)
        with open(os.path.join(trainloader.dataset.path_to_outputs, "model_weights/inputs_to_models.json"), "w") as fptr:
            fptr.write(json.dumps(inputs_to_model, indent=4))

        end_time = time.time()
        print(f"Finished Re-Training {args.improve_model_weights} in " + format_time(end_time - start_time))

    if(args.train):
        ### Training the entire tree
        # Download datasets
        train_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), train=True)
        test_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), train=False)

        # Dataset features
        class_labels = train_dataset.labels # List of unique classes in the dataset
        image_shape = train_dataset.image_shape # Get the shape of the image in the dataset (1xCxHxW)

        # Create dataloaders
        trainloader = get_dataloader(train_dataset, args.train_batch_size, args.num_workers, shuffle=True)
        testloader = get_dataloader(test_dataset, args.eval_batch_size, args.num_workers, shuffle=True)
        
        # Train supergroups
        hyperparameters = get_hyperparameters(trainloader)
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
            print(f"Training Started on Module {current_supergroup}")
            optimizer = torch.optim.Adam(list_of_models[-1].parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
            path_save_model = os.path.join(trainloader.dataset.path_to_outputs, f"model_weights/{current_supergroup}.pt")
            image_shape = train(list_of_models=list_of_models, current_supergroup=current_supergroup, epochs=hyperparameters['epochs'], optimizer=optimizer, model_save_path=path_save_model, trainloader=trainloader, validationloader=testloader)
            image_shape = tuple(image_shape[1:]) # change from (BxCxHxW) -> (CxHxW)

            # Create the average softmax of this current trained supergroup
            print("Computing Average SoftMax")
            list_of_models[-1].load_state_dict(torch.load(path_save_model)) # load the weights to the last model which is now trained
            path_to_softmax_matrix = os.path.join(trainloader.dataset.path_to_outputs, f"model_softmax/{current_supergroup}_avg_softmax.pt")
            AverageSoftmax(list_of_models, trainloader, current_supergroup, path_to_softmax_matrix)

            # Update the target_map based on the softmax of the current supergroup
            print("Updating TargetMap")
            path_decisions = trainloader.dataset.get_path_decisions() # the paths down the tree from the root node to each supergroup
            list_of_new_supergroups = update_target_map(trainloader, current_supergroup, hyperparameters['grouping_volatility'], path_to_softmax_matrix, path_decisions[current_supergroup], debug=args.debug)
            nodes_dict = trainloader.dataset.get_dictionary_of_nodes() # updated dictionary of nodes

            # If there are new supergroups then add it to the end of the queue and update the dictionary_of_inputs_for_models
            if(list_of_new_supergroups):
                supergroup_queue.extend(list_of_new_supergroups)
                if(args.debug):
                    print(f"List of Supergroups to still train: {supergroup_queue}")
            supergroup_queue = deque([sg for sg in supergroup_queue if sg in list(nodes_dict.keys())]) # update the queue to remove supergroups that have been grouped with other supergroups

            # Re-train the current supergroup on the number of children it has rather than the number of classes 
            node = nodes_dict[current_supergroup]
            num_children = node.num_groups # the number of children the node has
            num_classes = len(node.classes) # the number of categories from the dataset the node is responsible for
            
            print(f"For the current supergroup: {current_supergroup}, they have {num_classes} labels and {num_children} children with an image shape of {dictionary_of_inputs_for_models[current_supergroup][0]}")
            if(num_children != num_classes):
                # Then this node must be re-trained to only distinguish between the number of children nodes and not categories
                print(f"Re-training on the current supergroup {current_supergroup} with number of children: {num_children}")
                dictionary_of_inputs_for_models[current_supergroup][1] = num_children # changing the number of groups for the model to distinguish between
                list_of_models = get_list_of_models_by_path(dataloader=trainloader, model_backbone=args.model_backbone, current_supergroup=current_supergroup, dictionary_of_inputs_for_models=dictionary_of_inputs_for_models, debug_flag=args.debug) # reset the weights of the current supergroup so we don't load its state_dict
                
                optimizer = torch.optim.Adam(list_of_models[-1].parameters(), lr=hyperparameters["descendant_learning_rate"], weight_decay=hyperparameters["descendant_weight_decay"])
                image_shape = train(list_of_models=list_of_models, current_supergroup=current_supergroup, epochs=hyperparameters['epochs'], optimizer=optimizer, model_save_path=path_save_model, trainloader=trainloader, validationloader=testloader)
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
        test_dataset = GenerateDataset(args.dataset.lower(), args.model_backbone.lower(), train=False)
        testloader = get_dataloader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)
        
        conf = test(testloader)
        display_confusion_matrix(conf, testloader)

        end_time = time.time()
        print(f"Finished Testing TRUNK in " + format_time(end_time - start_time))

if __name__ == "__main__":
    main()

    """
    Notes
    Train: CUDA_LAUNCH_BLOCKING=1 python main.py --train --dataset emnist --model_backbone mobilenet --num_workers 2 --debug 
    Test: CUDA_LAUNCH_BLOCKING=1 python main.py --infer --dataset emnist --model_backbone mobilenet --num_workers 2 --debug
    """
