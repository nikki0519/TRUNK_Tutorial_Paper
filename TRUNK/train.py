# ----------------------------------------------------
# Name: train.py
# Purpose: Script to train the TRUNK network
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.optim as optim

## Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for train.py")
device = torch.device(device) # push the device to the gpu if gpu is available otherwise keep it on cpu

loss_function = nn.NLLLoss() # loss function used to compute loss during training and validation

def get_training_details(config, current_sg_model):
    """
    Get the scheduler function based on the config file

    Parameters
    ----------
    config: dict
        dictionary containing information on the hyperparameters and training regime

    current_sg_model: torch.nn.Module
        The current supergroup's network

    Return
    ------
    scheduler: torch.optim.lr_scheduler
    """
    epochs = config['epochs']
    optimizer_config = config.optimizer[0]
    scheduler_config = config.lr_scheduler[0]

    optimizer_type = optimizer_config['type']
    params = optimizer_config.get('params', {})
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(current_sg_model.parameters(), **params)

    scheduler_type = scheduler_config['type']
    params = scheduler_config.get('params', {})
    scheduler_class = getattr(lr_scheduler, scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler, optimizer, epochs


def train(list_of_models, current_supergroup, config, model_save_path, trainloader, validationloader):
    """
    Train the current supergroup module

    Parameters
    ----------
    list_of_models: list
        The list of all the supergroup models
    
    current_supergroup: str
        The current supergroup label
    
    epochs: int
        Number of epochs to train for

    config: dict
        dictionary containing information on the hyperparameters and training regime

    model_save_path: str
        the path to save the trained supergroup module
        
    trainloader: torch.utils.data.DataLoader
        The training dataloader to train the supergroup modules with

    validationloader: torch.utils.data.DataLoader
        The validation dataloader to conduct validation on the supergroup modules with

    Return
    ------
    feature_map_shape: tuple (BxCxHxW)
        the new shape of the feature map
    """
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    scheduler, optimizer, epochs = get_training_details(config, list_of_models[-1])
    max_validation_accuracy = 0.0 # keep track of the maximum accuracy to know which model to save after conducting validation
    for epoch in range(1, epochs+1):
        count = 0 # Count the number of times we get a true positive result in a batch, used to calculate accuracy
        total = 0 # Total number of values in a batch, used to calculate accuracy
        path_decisions = trainloader.dataset.get_path_decisions() # dictionary that stores the the path down the tree to a specific supergroup

        # Train Accuracy
        with tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{epochs}") as progress_bar:
            for batch_idx, (images_in_batch, target_maps_in_batch) in progress_bar:
                images_in_batch = images_in_batch.to(device) # The images in this current batch
                depth = 0 # The depth or layer of the tree we're currently at for a category
                current_node_in_batch = target_maps_in_batch[depth].to(device) # Get the current node or supergroup we're currently at in the tree for every image in the batch based on the category of that image
                indices_encountered = [] # Collect the list of indices of the images in the batch that have the corresponding correct next child based on the current supergroup model we're at
                noBatch = False # If none of the images in the batch have the right paths predicted, then set noBatch=True so that we can skip this batch

                """
                This loop in line 62 iterates through the list of models to identify whether current_node_in_batch which is given by target_map 
                is the same as the true node at the current depth of the path from root to the current supergroup as determined by path_decisions. 
                The current_node_in_batch and true node is updated by incrementing depth by one and model_idx corresponds with depth.
                """

                for model_idx in range(len(list_of_models) - 1): 
                    images_in_batch, _ = list_of_models[model_idx].evaluate(images_in_batch)
                    true_node_idx = path_decisions[current_supergroup][depth]
                    depth += 1

                    indices = torch.nonzero(current_node_in_batch == true_node_idx)[:,0] # identify all the images in the batch that have the same node as the node identified as by path_decisions at the current depth and record its indices
                    if(len(indices) > 0): # check if there are images whose predicted nodes from the target_map match the true node
                        images_in_batch = images_in_batch[indices] # only consider the images that have the right node indicated by target_map at this current depth 
                        new_indices = indices.cpu() 
                        for curr_depth in range(model_idx, 0, -1):
                            # this loop will iterate back to previous depths from the current depth to identify the images that have the right node at every depth and only preserve those images by recording only those indices
                            new_indices = indices_encountered[curr_depth - 1][new_indices]
                                            
                        indices_encountered.append(indices)
                        current_node_in_batch = target_maps_in_batch[depth][new_indices].to(device) # update this variable to only examine the images that have the right node at every depth of the path thus far

                    else:
                        noBatch = True # if no images in the batch are ground-truth images, then set noBatch to True so that we can skip this batch during training
                        break
                
                if(noBatch or images_in_batch.shape[0] == 0):
                    # skip this batch if no ground-truth images are identified for training and computing loss
                    continue

                list_of_models[depth].train() # train the current model at the latest depth of the tree
                optimizer.zero_grad() 
                images_in_batch, sg_prediction = list_of_models[depth](images_in_batch)
                loss = loss_function(sg_prediction, current_node_in_batch) # compute the loss given the loss function
                loss.backward()
                optimizer.step()
                scheduler.step()

                sg_prediction = sg_prediction.max(1, keepdim=True)[1] # get the supergroup index prediction 
                count += sg_prediction.eq(current_node_in_batch.view_as(sg_prediction)).sum().item() # reshape the current_node_in_batch to the shape of sg_prediction and count the number of nodes that are equal in the batch
                total += current_node_in_batch.shape[0] # total number of nodes in the batch at the current depth

                progress_bar.set_description(f"Epoch {epoch}/{epochs}, Batch {batch_idx + 1} / {len(trainloader)}, LR {scheduler.get_last_lr()[0]} - Train Loss: {loss / (batch_idx + 1)} | Train Acc: {100 * count / total} | Total Images Processed: {total}") # output the accuracy and training loss on the progress bar
            
        max_validation_accuracy, feature_map_size = validation(list_of_models, epoch, current_supergroup, max_validation_accuracy, model_save_path, validationloader)
    return feature_map_size

def validation(list_of_models, epoch, current_supergroup, max_validation_accuracy, model_save_path, validationloader):
    """
    Conduct validation testing on the current supergroup module

    Parameters
    ----------
    list_of_models: list
        The list of all the supergroup models

    epoch: int
        current epoch during training
    
    current_supergroup: str
        The current supergroup label
    
    max_validation_accuracy: float
        Keep track of the maximum accuracy to know which model to save after conducting validation

    model_save_path: str
        the path to save the trained supergroup module

    validationloader: torch.utils.data.DataLoader
        The validation dataloader to conduct validation on the supergroup modules with

    Return
    ------
    max_validation_accuracy: float
        the updated max_validation_accuracy if there is an update in the accuracy of the model
    
    images_in_batch.shape: tuple (BxCxHxW)
        the new shape of the feature map
    """

    count = 0
    total = 0
    path_decisions = validationloader.dataset.get_path_decisions()

    for images_in_batch, target_maps_in_batch in validationloader:
        images_in_batch = images_in_batch.to(device)
        noBatch = False
        depth = 0
        current_node_in_batch = target_maps_in_batch[depth].to(device)
        indices_encountered = []

        """
        This loop in line 123 iterates through the list of models to identify whether current_node_in_batch which is given by target_map 
        is the same as the true node at the current depth of the path from root to the current supergroup as determined by path_decisions. 
        The current_node_in_batch and true node is updated by incrementing depth by one and model_idx corresponds with depth.
        """
        
        for model_idx in range(len(list_of_models) - 1):
            images_in_batch, _ = list_of_models[model_idx].evaluate(images_in_batch)
            true_node_idx = path_decisions[current_supergroup][depth]
            depth += 1

            indices = torch.nonzero(current_node_in_batch == true_node_idx)[:,0] # identify all the images in the batch that have the same node as the node identified as by path_decisions at the current depth and record its indices 
            if(len(indices) > 0): # check if there are images whose predicted nodes from the target_map match the true node
                images_in_batch = images_in_batch[indices] # only consider the images that have the right node indicated by target_map at this current depth 
                new_indices = indices.cpu()
                for curr_depth in range(model_idx, 0, -1):
                    # this loop will iterate back to previous depths from the current depth to identify the images that have the right node at every depth and only preserve those images by recording only those indices
                    new_indices = indices_encountered[curr_depth - 1][new_indices]
                                    
                indices_encountered.append(indices)
                current_node_in_batch = target_maps_in_batch[depth][new_indices].to(device) # update this variable to only examine the images that have the right node at every depth of the path thus far

            else:
                noBatch = True # if no images in the batch are ground-truth images, then set noBatch to True so that we can skip this batch during validation
                break

        if(noBatch or images_in_batch.shape[0] == 0):
            # skip this batch if no ground-truth images are identified for validation and computing loss
            continue
        
        images_in_batch, sg_prediction = list_of_models[depth](images_in_batch)
        sg_prediction = sg_prediction.max(1, keepdim=True)[1] # get the supergroup index prediction 
        count += sg_prediction.eq(current_node_in_batch.view_as(sg_prediction)).sum().item() # reshape the current_node_in_batch to the shape of sg_prediction and count the number of nodes that are equal in the batch
        total += current_node_in_batch.shape[0] # total number of nodes in the batch at the current depth

    if(total > 0):
        print(f"Validation Accuracy for supergroup {current_supergroup} at epoch {epoch}: {count / total * 100}")
        if(count / total > max_validation_accuracy):
            max_validation_accuracy = count / total
            torch.save(list_of_models[-1].state_dict(), model_save_path)

    return max_validation_accuracy, images_in_batch.shape