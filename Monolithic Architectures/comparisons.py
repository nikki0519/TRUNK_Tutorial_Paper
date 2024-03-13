# ----------------------------------------------------
# Name: comparisons.py
# Purpose: Script to execute the necessary function calls based on the user's needs
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
from torchvision import datasets
import torchvision.transforms as tvt
import torch
import argparse
from ResNet.resnet import resnet
from VGG.vgg import vgg
from ConvNeXt.convnext import convnext
from DinoV2.dinov2 import dinov2
from DeIT.deit import deit
from MobileNet.mobilenetv2 import mobilenet
from ViT.vit import vit
from ResNetQuantized.resnetQuantized import resnet_quantized

def parser():
	"""
	Get command-line arguments

	Return
	------
	args: argparse.Namespace
		user arguments 
	"""
	parser = argparse.ArgumentParser(description="Comparing other Image Classification Models")
	parser.add_argument("--train_batch_size", type=int, help="Training Batch Size", default=16)
	parser.add_argument("--eval_batch_size", type=int, help="Evaluation Batch Size", default=1)
	parser.add_argument("--num_workers", type=int, help="Number of parallel workers for dataloader", default=0)
	parser.add_argument("--dataset", type=str, help="EMNIST, SVHN, CIFAR-10", choices=["emnist", "svhn", "cifar10"], default="emnist")
	parser.add_argument("--model", type=str, help="choose between resnet, vgg, resnet_quantized, mobilenet, convnet, vit, dinov2, deit", choices=["resnet", "vgg", "resnet_quantized", "mobilenet", "convnext", "vit", "dinov2", "deit"], default="resnet")
	args = parser.parse_args()
	return args
	
def load_dataset(dataset, train=False):
	"""
	Return a torchvision dataset given user input of which dataset they want to conduct image classification for using TRUNK

	Parameters
	----------
	dataset: str
		the user's choice of dataset (i.e. emnist, svhn, or cifar10) inputted as a command line argument

	train: bool
		train is true, if we want to create a training dataset and train is false if we want to create a testing dataset
	
	Return
	------
	data: torchvision.dataset
		the torchvision dataset that the user wants to train/test on using TRUNK
	"""
	
	transform = [tvt.Resize(size=(224, 224)), tvt.ToTensor()] # conduct transformation on the image to convert it to size 224x224 and torch.tensor type
	if(dataset != "emnist"):
		transform.append(tvt.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])) # only the emnist dataset cannot be normalized so we omit this transformation for the emnist dataset
	transform = tvt.Compose(transform)
	
	if(train):
		if(dataset == "emnist"):
				return datasets.EMNIST(
								root="/scratch/gilbreth/ravi30/data/train",
								split="balanced",
								train=True,
								download=True,
								transform=transform
						)
		
		elif(dataset == "svhn"):
			return datasets.SVHN(root="/scratch/gilbreth/ravi30/data/train/",
								split="train",
								download=True,
								transform=transform)
		
		elif(dataset == "cifar10"):
			return datasets.CIFAR10(root="/scratch/gilbreth/ravi30/data/train/", 
									 train=True, 
									 download=True, 
									 transform=transform)
		 
	else:
		if(dataset == "emnist"):
			return datasets.EMNIST(
							root="/scratch/gilbreth/ravi30/data/test/",
							split="balanced",
							train=False,
							download=True,
							transform=transform)

		elif(dataset == "svhn"):
			return datasets.SVHN(root="/scratch/gilbreth/ravi30/data/test/",
								split="test",
								download=True,
								transform=transform)
		
		elif(dataset == "cifar10"):
			return datasets.CIFAR10(root="/scratch/gilbreth/ravi30/data/test/", 
									train=False, 
									download=True, 
									transform=transform)

def get_dataloader(dataset, batch_size, num_workers, shuffle):
	"""
	Return an iterable torch dataloader

	Parameters
	----------
	dataset: torchvision.dataset
		the torchvision dataset that the user wants to train/test on using TRUNK

	batch_size: int
		batch size of the dataloader
	
	num_workers: int
		parallel workers for the dataloader
	
	shuffle: bool
		shuffle the dataset in the dataloader

	Return
	------
	dataloader: torch.utils.data.DataLoader
		return the iterable dataloader for the custom dataset	
	"""

	if(len(dataset) % batch_size != 0):
		return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True)
	
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def main():
	args = parser()

	# Load the datasets
	train_dataset = load_dataset(args.dataset, train=True)
	test_dataset = load_dataset(args.dataset, train=False)

	# Get the dataloaders
	trainloader = get_dataloader(train_dataset, args.train_batch_size, args.num_workers, shuffle=True)
	testloader = get_dataloader(test_dataset, args.eval_batch_size, args.num_workers, shuffle=True)

	# Execute the chosen model
	model_dictionary = {"resnet": resnet, "resnet_quantized": resnet_quantized, "vgg": vgg, "convnext": convnext, "dinov2": dinov2, "mobilenet": mobilenet, "vit": vit, "deit": deit}
	model_dictionary[args.model.lower()](trainloader, testloader, args.dataset) 


if __name__ == "__main__":
	main()
