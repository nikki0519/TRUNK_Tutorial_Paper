# Import necessary libraries
from torchvision import datasets
from torchvision import transforms
import torch
import json
import os
from pathDecisions import get_path_decisions, get_paths, get_num_classes_per_sg, update_tree, load_tree_from_file, get_leaf_nodes, update_tree_attributes

class GenerateDataset(torch.utils.data.Dataset):
	"""
	Create a custom dataset for the purposes of image classification using TRUNK

	Attributes
	----------
	dataset: str
		the user's choice of dataset (i.e. emnist, svhn, or cifar10) inputted as a command line argument

	model_backbone: str
		the user's choice of model (i.e. vgg or mobilenet) inputted as a command line argument

	data: torchvision.dataset
		the torchvision dataset we are using to train/test our TRUNK model on

	max_depth: int
		current maximum depth of the tree 

	image_shape: tuple
		shape of the image in the dataset in the format (channels x height x width)

	labels: list
		list of all the unique classes in the dataset

	path_to_outputs: str
		the path to where we store all our outputs (i.e. target_map, path_decisions, etc.)


	Methods
	-------
	get_target_map()
		get the updated target_map that is saved in a json file

	get_path_decisions()
		get an updated path_decisions (dictionary that maps the path from the root to each supergroup in the tree) based on the new target_map by index positions

	get_paths()
		get an updated paths (dictionary that maps the path from the root to each supergroup in the tree) based on the new target_map by the values

	get_num_classes_per_sg()
		Get a dictionary mapping of the supergroups and the number of categories each supergroup is responsible for
		
	update_target_map(target_map)
		write the updated target_map to the json file it was originally saved in

	get_integer_encoding()
		get a dictionary of labels and their respective class IDs

	get_inverse_integer_encoding()
		get a dictionary of class IDs and their respective labels
	"""

	def __init__(self, dataset, model_backbone, config, train=False, re_train=False):
		"""
		Parameters
		----------
		dataset: str
			the user's choice of dataset (i.e. emnist, svhn, or cifar10) inputted as a command line argument

		model_backbone: str
			the user's choice of model_backbone (i.e. vgg or mobilenet) inputted as a command line argument

		config: dict
			dictionary of training regime

		train: bool
			train is true, if we want to create a training dataset and train is false if we want to create a testing dataset

		re_train: bool
			if re_train, then we want to re-train a specific module in the tree
		"""

		self.dataset = dataset # Argument from user: emnist or svhn or cifar10?
		self.model_backbone = model_backbone # Argument from user: vgg or mobilenet, needed for path
		self.data = load_dataset(self.dataset, config, train) # Load the torchvision dataset
		self.max_depth = 1 # Current maximum depth of the tree 

		self.image_shape = tuple(self.data[0][0].shape) # Get the shape of the image in the dataset (CxHxW)
		if(self.dataset == "svhn"):
			self.labels = list(set(self.data.labels))
		else:
			self.labels = self.data.classes # List of unique classes in the dataset

		self.path_to_outputs = os.path.join("./Datasets", self.dataset) # Path to save outputs for particular dataset used
		self.path_to_outputs = os.path.join(self.path_to_outputs, self.model_backbone)

		if(train and not re_train):
			# Initialize a target_map and record it (description of target_map in the get_target_map function)
			target_map = {idx: [idx] for idx in range(len(self.labels))}
			path_to_target_map = os.path.join(self.path_to_outputs, "target_map.json") # Path to save target map
			with open(path_to_target_map, "w") as fptr:
				fptr.write(json.dumps(target_map, indent=4))

			# Initialize and save the tree structure (description of tree is in pathDecisions.py)
			self.update_tree()
		
	def get_target_map(self):
		"""
		get the updated target_map (dictionary that maps the path from the root to each classID) that is saved in a json file

		what is a target_map? Here is an example of TRUNK tree:
		
		     |root|
			/       \
		|node 0|      |node 1|
		/     \       /        \
	|node 0| |node 1| |node 0|  |node 1|
		
		If the target_map is as follows: target_map = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
		then for the classID 2, the path down the tree is from root -> 1st child of root -> 0th child of the 1st child of root and so on so forth

		Return
		------
		target_map: dict
			dictionary that maps the path from the root to each class label
		"""

		path_to_target_map = os.path.join(self.path_to_outputs, "target_map.json") # Path to save target map
		fptr = open(path_to_target_map, "r")
		return json.load(fptr)
	
	def get_inverse_target_map(self):
		"""
		Get the inverse target_map where the key is the path to the leaf and the value is the category

		Return
		------
		inverse_target_map: dict
			dictionary that maps the category by the path from the root to leaf
		"""

		def remove_padding_from_path(path):
			"""
			Remove the -1s in the path to ignore the padding

			Parameters
			----------
			path: list
				the path down to the supergroup

			Return
			------
			path: list
				path without padding
			"""

			reversed_path = path[::-1]
			for idx, value in enumerate(reversed_path):
				if(value != -1):
					return reversed_path[idx:][::-1]
			
			return []

		target_map_with_padding = self.get_target_map()
		target_map = {}
		for category, path_to_category in target_map_with_padding.items():
			target_map[category] = remove_padding_from_path(path_to_category)

		return {tuple(path_to_leaf): category for category, path_to_leaf in target_map.items()}
	
	def get_dictionary_of_nodes(self):
		"""
		Get the dictionary of all the nodes (i.e. {"root": root_node, "sg1": sg1_node, ...})

		Return
		------
		nodes_dict: dict
			dictionary of nodes
		"""

		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		nodes_dict = load_tree_from_file(path_to_tree)
		return nodes_dict
	
	def get_path_decisions(self):
		"""
		get an updated path_decisions (dictionary that maps the path from the root to each supergroup in the tree) based on the new target_map by index positions

		what is path_decisions? Here is an example of TRUNK tree:

			 |root|
			/     \
		|SG1|     |SG4|
		/   \       /  \
	|SG2| |SG3| |SG5|  |SG6|

		If the path_decision is as follows: path_decisions = {root: [], SG1: [0], SG2: [0, 0], SG3: [0, 1], SG4: [1], SG5: [1, 0], SG6: [1, 1]}
		then for SG5, the path down the tree is from root -> 1st child of root -> 0th child of the 1st child of root and so on so forth

		Return
		------
		path_decisions: dict
			dictionary that maps the path from the root to each supergroup in the tree
		"""

		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		return get_path_decisions(path_to_tree)
	
	def get_inverse_path_decisions(self):
		"""
		Get the inverse path_decisions where the key is the path to the supergroup and the value is the supergroup name

		Return
		------
		inverse_path_decisions: dict
			dictionary that maps the sg by the path from the root to sg
		"""

		path_decisions = self.get_path_decisions()
		return {tuple(path_to_sg): sg for sg, path_to_sg in path_decisions.items()}
	
	def get_paths(self):
		"""
		get an updated paths (dictionary that maps the path from the root to each supergroup in the tree) based on the new target_map by the values

		what is paths? Here is an example of TRUNK tree:

			 |root|
			/     \
		|SG1|     |SG4|
		/   \       /  \
	|SG2| |SG3| |SG5|  |SG6|

		If the path is as follows: path = {root: [root], SG1: [root, SG1], SG2: [root, SG1, SG3], SG3: [root, SG1, SG3], SG4: [root, SG4], SG5: [root, SG4, SG5], SG6: [root, SG4, SG6]}
		then for SG5, the path down the tree is from root -> SG4 -> SG5 and so forth

		Return
		------
		paths: dict
			dictionary that maps the path from the root to each supergroup in the tree by the value of nodes
		"""

		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		return get_paths(path_to_tree)
	
	def get_num_classes_per_sg(self):
		"""
		Get a dictionary mapping of the supergroups and the number of categories each supergroup is responsible for
		Example: root manages all the categories of the dataset (i.e. 9) but SG1 only examines categories from the dataset that are visually similar (i.e. 5)

		Return
		------
		num_class_per_sg: dict
        	dictionary mapping each node and the number of classes they are responsible for
		"""

		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		return get_num_classes_per_sg(path_to_tree)
	
	def get_leaf_nodes(self):
		"""
		Get a dictionary of nodes that are the leaves of the tree and its respective paths from the root

		Return
		------
		leafs: dict
			dictionary of leaf nodes and its respective path from the root node
    	"""

		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		return get_leaf_nodes(path_to_tree)
	
	def update_target_map(self, target_map):
		"""
		write the updated target_map to the json file it was originally saved in

		Parameters
		----------
		target_map: dict
			the updated dictionary that maps the path down the tree from the root node to its respective classID leafs
		"""

		path_to_target_map = os.path.join(self.path_to_outputs, "target_map.json") # Path to save target map
		# nodes_dict = self.get_dictionary_of_nodes()
		# self.max_depth = nodes_dict["root"].depth_of_tree()

		# padded_target_map = self.padding_target_map(target_map)
		with open(path_to_target_map, "w") as fptr:
			fptr.write(json.dumps(target_map, indent=4))

	def padding_target_map(self):
		"""
		Pad the paths using a -1 value in the target_map to the maximum depth in the tree so that the dataset items have uniform length 

		"""
		target_map = self.get_target_map()
		nodes_dict = self.get_dictionary_of_nodes()
		self.max_depth = nodes_dict["root"].depth_of_tree()

		for category, path_to_category in target_map.items():
			if(len(path_to_category) < self.max_depth):
				for jdx in range(self.max_depth - len(path_to_category)):
					target_map[category].append(-1)

		self.update_target_map(target_map)

	def update_tree(self, current_supergroup=None):
		"""
		write the updated tree to the pickle file it was originally saved in

		Parameters
		----------
		past_nodes_dict: dict [Optional]
			previous dictionary of nodes

		current_supergroup: str [Optional]
			the current supergroup we're at and is trained so we want to update that 
		Return
		------
		"""		
		
		target_map = self.get_target_map()
		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")

		path_to_weights = None
		if(current_supergroup):
			path_to_weights = os.path.join(self.path_to_outputs, f"{current_supergroup}_supergroups.json")

		update_tree(path_to_tree, target_map, current_supergroup, path_to_weights)

	def update_tree_attributes(self, nodes_dict):
		"""
		Update the attributes of the tree

		Parameters
		----------
		nodes_dict: dict
			dictionary of nodes
		"""
		
		path_to_tree = os.path.join(self.path_to_outputs, "tree.pkl")
		update_tree_attributes(path_to_tree, nodes_dict)

	def get_integer_encoding(self):
		"""
		get a dictionary of labels and their respective class IDs

		Return
		------
		integer_encoding: dict
			dictionary that maps each class label with its respective classID
		"""

		return {self.labels[idx]: idx for idx in range(len(self.labels))}
	
	def get_inverse_integer_encoding(self):
		"""
		get a dictionary of classIDs and their respective class labels

		Return
		------
		inverse_integer_encoding: dict
			dictionary that maps each classID with its respective class labels
		"""

		return {idx: self.labels[idx] for idx in range(len(self.labels))}

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		"""
		Parameters
		----------
		idx: int
			index in the dataset

		Return
		------
		image: torch.Tensor
			the idx-th image in the dataset

		target: list
			the path from the root to the image's class leaf in the tree
		"""

		self.target_map = self.get_target_map() # get the updated and most recent target_map stored in the json file
		image, target = self.data[idx][0], self.target_map[str(self.data[idx][1])] # Note: self.data[idx] = (image, classID)
		return image, target

def build_transforms(transform_config):
	"""
	Build the list of transforms appropriate to the dataset

	Parameters
	----------
	transform_config: dict
		dictionary of transforms we're using

	Return
	------
	transform_list: list
		list of transforms
	"""

	transform_list = []
	for item in transform_config:
		transform_type = item['type']
		params = item.get('params', {})
		transform_class = getattr(transforms, transform_type)
		transform_list.append(transform_class(**params))

	return transforms.Compose(transform_list)

def load_dataset(dataset, config, train=False):
	"""
	Return a torchvision dataset given user input of which dataset they want to conduct image classification for using TRUNK

	Parameters
	----------
	dataset: str
		the user's choice of dataset (i.e. emnist, svhn, or cifar10) inputted as a command line argument

	config: dict
		dictionary of training regime

	train: bool
		train is true, if we want to create a training dataset and train is false if we want to create a testing dataset
	
	Return
	------
	data: torchvision.dataset
		the torchvision dataset that the user wants to train/test on using TRUNK
	"""
	
	transform_config = config.dataset.train.transform
	transform = build_transforms(transform_config)
	
	if(train):
		if(dataset == "emnist"):
				return datasets.EMNIST(
								root="../data/train",
								split="balanced",
								train=True,
								download=True,
								transform=transform
						)
		
		elif(dataset == "svhn"):
				return datasets.SVHN(root="../data/train/",
								split="train",
								download=True,
								transform=transform)
		
		elif(dataset == "cifar10"):
			return datasets.CIFAR10(root="../data/train/", 
									 train=True, 
									 download=True, 
									 transform=transform)
		 
	else:
		if(dataset == "emnist"):
			return datasets.EMNIST(
							root="../data/test/",
							split="balanced",
							train=False,
							download=True,
							transform=transform)

		elif(dataset == "svhn"):
			return datasets.SVHN(root="../data/test/",
								split="test",
								download=True,
								transform=transform)
		
		elif(dataset == "cifar10"):
			return datasets.CIFAR10(root="../data/test/", 
									train=False, 
									download=True, 
									transform=transform)

def get_dataloader(dataset, config, train=False, validation=False):
	"""
	Return an iterable torch dataloader

	Parameters
	----------
	dataset: torchvision.dataset
		the torchvision dataset that the user wants to train/test on using TRUNK

	config: dict
		dictionary of training regime

	train: bool [Optional]
		train is False if we are doing inference else it is True

	validation: bool [Optional]
		validation is true if training is true and we need a validation dataset

	Return
	------
	dataloader: torch.utils.data.DataLoader
		return the iterable dataloader for the custom dataset	
	"""

	if(train):
		if(validation):
			dataloader_config = config.dataset.validation.params
		else:
			dataloader_config = config.dataset.train.params
	else:
		dataloader_config = config.dataset.test.params

	return torch.utils.data.DataLoader(dataset, batch_size=dataloader_config["batch_size"], num_workers=dataloader_config["num_workers"], shuffle=dataloader_config["shuffle"], drop_last=True)