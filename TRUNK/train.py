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
import os

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
    return start_epoch

def train(list_of_models, current_supergroup, config, grouping_volatility, model_save_path, trainloader, validationloader):
    """
    Train the current supergroup module

    Parameters
    ----------
    list_of_models: list
    current_supergroup: str
    epochs: int
    config: OmegaConf
        dictionary containing information on the hyperparameters and training regime
    grouping_volatility: float
        the factor used in the ASL calculation and plays a role in how the tree is structured
    model_save_path: str
    trainloader: torch.utils.data.DataLoader
    validationloader: torch.utils.data.DataLoader

    Return
    ------
    feature_map_shape: tuple (BxCxHxW)
    """
    scheduler, optimizer, loss_function, epochs = get_training_details(config, list_of_models[-1])
    # Log metrics on Weights and Biases Platform
    run = wandb.init(
        project=f"TRUNK-{trainloader.dataset.dataset}",
        config={
            "architecture": current_supergroup,
            "backbone": trainloader.dataset.model_backbone,
            "dataset": trainloader.dataset.dataset,
            "grouping_volatility": grouping_volatility,
            "epochs": epochs,
            "learning_rate": config.optimizer[0].params.lr,
            "weight_decay": config.optimizer[0].params.weight_decay,
            "lr_scheduler": config.lr_scheduler[0].type
        }
    )

    start_epoch = 1
    if(os.path.exists(model_save_path)):
        start_epoch = load_checkpoint(list_of_models[-1], optimizer, model_save_path)

    max_validation_accuracy = 0.0 # keep track of the maximum accuracy to know which model to save after conducting validation
    for epoch in range(start_epoch, epochs+1):
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
                This loop in line 62 iterates through the list of models to identify all the images that have the correct predictions across all the nodes 
                down the tree until we reach the current node we intend to train. This will be our set of ground-truth images for the current node. 
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

                sg_prediction = sg_prediction.max(1, keepdim=True)[1] 
                count += sg_prediction.eq(current_node_in_batch.view_as(sg_prediction)).sum().item() 
                total += current_node_in_batch.shape[0] 
                running_training_loss += loss

                progress_bar.set_description(f"Epoch {epoch}/{epochs}, Batch {batch_idx + 1} / {len(trainloader)}, LR {scheduler.get_last_lr()[0]} - Train Loss: {loss / (batch_idx + 1)} | Train Acc: {100 * count / total} | Total Images Processed: {total}") # output the accuracy and training loss on the progress bar    
        max_validation_accuracy, validation_accuracy, feature_map_size = validation(list_of_models, optimizer, epoch, current_supergroup, max_validation_accuracy, model_save_path, validationloader)
        scheduler.step()
        wandb.log({"train loss": running_training_loss / len(trainloader), "validation accuracy": validation_accuracy, "lr": optimizer.param_groups[0]["lr"]})

    wandb.finish()
    return feature_map_size

def validation(list_of_models, optimizer, epoch, current_supergroup, max_validation_accuracy, model_save_path, validationloader):
    """
    Conduct validation testing on the current supergroup module

    Parameters
    ----------
    list_of_models: list
    optimizer: torch.optim
    epoch: int
    current_supergroup: str
    max_validation_accuracy: float
        Keep track of the maximum accuracy to know which model to save after conducting validation
    model_save_path: str
    validationloader: torch.utils.data.DataLoader

    Return
    ------
    max_validation_accuracy: float
        the updated max_validation_accuracy if there is an update in the accuracy of the model
    validation_accuracy: float
    images_in_batch.shape: tuple (BxCxHxW)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': list_of_models[-1].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)

    return max_validation_accuracy, validation_accuray, images_in_batch.shape