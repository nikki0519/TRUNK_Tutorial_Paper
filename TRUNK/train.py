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
import wandb

## Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for train.py")
device = torch.device(device)

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
    optimizer: torch.optim
    loss_function: torch.nn
    epochs: int
    """
    epochs = config['epochs']
    optimizer_config = config.optimizer[0]
    scheduler_config = config.lr_scheduler[0]
    loss_config = config.loss[0]

    optimizer_type = optimizer_config['type']
    params = optimizer_config.get('params', {})
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(current_sg_model.parameters(), **params)

    scheduler_type = scheduler_config['type']
    params = scheduler_config.get('params', {})
    scheduler_class = getattr(lr_scheduler, scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    loss_type = loss_config['type']
    params = loss_config.get('params', {})
    loss_class = getattr(nn, loss_type)
    loss_function = loss_class(**params)

    return scheduler, optimizer, loss_function, epochs


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
    scheduler, optimizer, loss_function, epochs = get_training_details(config, list_of_models[-1])
    # Log metrics on Weights and Biases Platform
    run = wandb.init(
        project="TRUNK",
        config={
            "architecture": current_supergroup,
            "dataset": trainloader.dataset.dataset,
            "epochs": epochs,
            "learning_rate": config.optimizer[0].params.lr,
            "weight_decay": config.optimizer[0].params.weight_decay,
            "lr_scheduler": config.lr_scheduler[0].type
        }
    )

    max_validation_accuracy = 0.0 # keep track of the maximum accuracy to know which model to save after conducting validation
    for epoch in range(1, epochs+1):
        running_training_loss = 0.0
        count = 0 
        total = 0 
        path_decisions = trainloader.dataset.get_path_decisions()

        # Train Accuracy
        with tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{epochs}") as progress_bar:
            for batch_idx, (images_in_batch, target_maps_in_batch) in progress_bar:
                images_in_batch = images_in_batch.to(device) 
                depth = 0 # The depth or layer of the tree we're currently at for a category
                current_node_in_batch = target_maps_in_batch[depth].to(device) 
                indices_encountered = [] # Collect the list of indices of the images in the batch that have the corresponding correct next child based on the current supergroup model we're at
                noBatch = False # If none of the images in the batch have the right paths predicted, then set noBatch=True so that we can skip this batch

                """
                This loop in line 62 iterates through the list of models to identify whether the current_node_in_batch, predicted by the models, 
                is the same as the true node at the current depth of the tree. Only those images that correspond to the true node at every depth of the tree
                is saved. This will be our ground truth. 
                """

                for model_idx in range(len(list_of_models) - 1): 
                    images_in_batch, _ = list_of_models[model_idx].evaluate(images_in_batch)
                    true_node_idx = path_decisions[current_supergroup][depth]
                    depth += 1

                    indices = torch.nonzero(current_node_in_batch == true_node_idx)[:,0] # identify all the images in the batch that have the same node as the node identified as by path_decisions at the current depth and record its indices
                    if(len(indices) > 0): 
                        images_in_batch = images_in_batch[indices] # only consider the images that have the right node 
                        new_indices = indices.cpu() 
                        for curr_depth in range(model_idx, 0, -1):
                            # this loop will iterate back to previous depths from the current depth to identify the images that have the right node at every depth and only preserve those images by recording only those indices
                            new_indices = indices_encountered[curr_depth - 1][new_indices].cpu()
                                            
                        indices_encountered.append(indices)
                        current_node_in_batch = target_maps_in_batch[depth][new_indices].to(device) 

                    else:
                        noBatch = True # if no images in the batch are ground-truth images, then set noBatch to True so that we can skip this batch during training
                        break
                
                if(noBatch or images_in_batch.shape[0] == 0):
                    # skip this batch if no ground-truth images are identified for training and computing loss
                    continue

                list_of_models[depth].train() # train the current model at the latest depth of the tree
                optimizer.zero_grad() 
                images_in_batch, sg_prediction = list_of_models[depth](images_in_batch)
                loss = loss_function(sg_prediction, current_node_in_batch) 
                loss.backward()
                optimizer.step()
                scheduler.step()

                sg_prediction = sg_prediction.max(1, keepdim=True)[1] 
                count += sg_prediction.eq(current_node_in_batch.view_as(sg_prediction)).sum().item() 
                total += current_node_in_batch.shape[0] 
                running_training_loss += loss

                progress_bar.set_description(f"Epoch {epoch}/{epochs}, Batch {batch_idx + 1} / {len(trainloader)}, LR {scheduler.get_last_lr()[0]} - Train Loss: {loss / (batch_idx + 1)} | Train Acc: {100 * count / total} | Total Images Processed: {total}") # output the accuracy and training loss on the progress bar    
        max_validation_accuracy, validation_accuracy, feature_map_size = validation(list_of_models, epoch, current_supergroup, max_validation_accuracy, model_save_path, validationloader)
        wandb.log({"train loss": running_training_loss / len(trainloader), "validation accuracy": validation_accuracy, "lr": optimizer.param_groups[0]["lr"]})
    
    wandb.finish()
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

    validation_accuracy: float
        the current epoch's validation accuracy
    
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
        
        for model_idx in range(len(list_of_models) - 1):
            images_in_batch, _ = list_of_models[model_idx].evaluate(images_in_batch)
            true_node_idx = path_decisions[current_supergroup][depth]
            depth += 1

            indices = torch.nonzero(current_node_in_batch == true_node_idx)[:,0] 
            if(len(indices) > 0): 
                images_in_batch = images_in_batch[indices] 
                new_indices = indices.cpu()
                for curr_depth in range(model_idx, 0, -1):
                    new_indices = indices_encountered[curr_depth - 1][new_indices].cpu()
                                    
                indices_encountered.append(indices)
                current_node_in_batch = target_maps_in_batch[depth][new_indices].to(device)

            else:
                noBatch = True
                break

        if(noBatch or images_in_batch.shape[0] == 0):
            continue
        
        images_in_batch, sg_prediction = list_of_models[depth](images_in_batch)
        sg_prediction = sg_prediction.max(1, keepdim=True)[1] 
        count += sg_prediction.eq(current_node_in_batch.view_as(sg_prediction)).sum().item() 
        total += current_node_in_batch.shape[0] 

    if(total > 0):
        validation_accuray = count / total * 100
        print(f"Validation Accuracy for supergroup {current_supergroup} at epoch {epoch}: {validation_accuray}")
        if(count / total > max_validation_accuracy):
            max_validation_accuracy = count / total
            torch.save(list_of_models[-1].state_dict(), model_save_path)

    return max_validation_accuracy, validation_accuray, images_in_batch.shape