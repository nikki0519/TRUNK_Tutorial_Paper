# ----------------------------------------------------
# Name: grouper.py
# Purpose: Script to group the similar categories
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torch
import torch.nn as nn
import math
import numpy as np
from collections import defaultdict
import json
import os
import copy

# Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device} for grouper.py")
device = torch.device(device)

def AverageSoftmax(list_of_models, dataloader, current_supergroup, softmax_file_path=None):
	"""
	Calculate the average softmax between each group to identify if the groups should be divided even more

	Parameters
	----------	
	list_of_models: list
		list of all the supergroup models

	dataloader: torch.utils.data.DataLoader
		the iterable dataloader for training purposes
	
	current_supergroup: str
		the current supergroup we're at in the tree

	softmax_file_path: str [Optional]
		the path to the softmax file
	"""

	path_decisions = dataloader.dataset.get_path_decisions() 
	number_of_classes_in_dataset = len(dataloader.dataset.labels)
	number_of_classes_grouped = list_of_models[-1].number_of_classes 

	softmax_matrix = torch.zeros((number_of_classes_grouped, number_of_classes_grouped)).to(device) # softmax matrix that will record the softmax values among the different supergroups
	counts = [0 for idx in range(number_of_classes_grouped)]
	with torch.no_grad():
		for batch_idx, (images_in_batch, target_maps_in_batch) in enumerate(dataloader):
			images_in_batch = images_in_batch.to(device) 
			depth = 0 # The depth of the tree we're currently at for a category
			current_node_in_batch = target_maps_in_batch[depth].to(device) 
			indices_encountered = [] 
			noBatch = False 

			for model_idx in range(len(list_of_models) - 1):
				images_in_batch, _ = list_of_models[model_idx].evaluate(images_in_batch)
				true_node_idx = path_decisions[current_supergroup][depth]
				depth += 1

				indices = torch.nonzero(current_node_in_batch == true_node_idx)[:,0]  
				if(len(indices) > 0): 
					new_indices = indices.cpu()
					for curr_depth in range(model_idx, 0, -1):
						new_indices = indices_encountered[curr_depth - 1][new_indices].cpu()
					
					indices_encountered.append(indices)
					current_node_in_batch = target_maps_in_batch[depth][new_indices].to(device)
					images_in_batch = images_in_batch[indices]
				
				else:
					noBatch = True
					break

			if(noBatch or images_in_batch.shape[0] == 0):
				continue
			
			images_in_batch, predictions = list_of_models[depth](images_in_batch) 
			predictions = nn.functional.softmax(predictions, dim=1) 
			for supergroup in range(number_of_classes_grouped):
				indices = torch.nonzero(current_node_in_batch == supergroup)[:, 0] # gather the indices of the images in the batch that have the same supergroup index
				hold = predictions[indices] # only keep the predictions that were the same
				softmax_matrix[supergroup] += hold.sum(dim=0) # add the total softmax of all the same predictions to the softmax group for the current supergroup
				counts[supergroup] += hold.shape[0] 

	for idx in range(number_of_classes_grouped):
		if(counts[idx] != 0):
			softmax_matrix[idx] /= counts[idx]
		else:
			softmax_matrix[idx] = 0

	if(softmax_file_path):
		torch.save(softmax_matrix, softmax_file_path)

def transitive_grouper(similar_groups):
	"""
	Once you have created a list of similar groups. This function returns a disjoint set of similar groups.

	Parameters
	----------
	similar_groups: list
		list of similar groups

	Return
	------
	similar_groups_dict: defaultdict
		disjoint set of similar groups
	"""

	similar_groups_dict = defaultdict(list)
	for group in similar_groups:
		for category1 in group:
			for category2 in group:
				if(category2 not in similar_groups_dict[category1]):
					similar_groups_dict[category1].append(category2)
				if(category1 not in similar_groups_dict[category2]):
					similar_groups_dict[category2].append(category1)
					
	# If category "A" is similar to category "B" and category "B" is similar to category "C". Then category "A" and "C" should be grouped together along with "B" into same SG in the hierarchy.
	for category in similar_groups_dict:
		for similar_category in similar_groups_dict[category]:
			for similar_to_similar_category in similar_groups_dict[similar_category]:
				if(similar_to_similar_category not in similar_groups_dict[category]):
					similar_groups_dict[category].append(similar_to_similar_category)
	return similar_groups_dict

def SigmoidMembership(number_of_classes, grouping_volatility, x):
	"""
	Sigmoid membership - fuzzy logic! Used to determine visual similarity

	Parameters
	----------
	number_of_classes: int
		the number of supergroups/children the node is responsible for

	grouping_volatility: float
		constant that contributes to the structure of the tree

	x: float
		the average softmax at a particular index (class idx, class jdx) of the average softmax matrix

	Return
	------
	sigmoidMembership: float
		determine the visual simularity between two classes
	"""
	
	return 1.0 / (1.0 + math.exp(-(x - (1.0 / number_of_classes * grouping_volatility)) / (1.0 / (10.0 * number_of_classes * grouping_volatility))))

def create_supergroup_list_helper(number_of_classes, grouping_volatility, avg_softmax_matrix):
	"""
	A helper function to creating a supergroup list

	Parameters
	----------
	number_of_classes: int
		number of supergroups we are examining

	grouping_volatility: float
		constant for sigmoidal membership
	
	avg_softmax_matrix: torch.tensor
		average softmax matrix of the visual similarities between each supergroup

	Return
	------
	similar_groups: list
		list of similar groups
	"""

	similar_groups = []
	for idx in range(number_of_classes):
		similar_groups.append([idx])
		for jdx in range(idx, number_of_classes):
			prob = SigmoidMembership(number_of_classes, grouping_volatility, avg_softmax_matrix[idx][jdx])
			# Determine if the expected value that two classes are visually similar is greater than 50%
			if((np.random.choice(2, 100000, p = [1 - prob, prob]).mean() > 0.5) and (idx != jdx)):
				similar_groups[idx].append(jdx)

	return similar_groups

def create_supergroup_list(number_of_classes, grouping_volatility, avg_softmax_matrix):
	"""
	create a list of supergroups for the current supergroup/node we're at. Each supergroup in the list is a list containing the supergroup IDs

	Parameters
	----------
	number_of_classes: int
		number of supergroups

	grouping_volatility: float
		constant for sigmoidal membership

	avg_softmax_matrix: torch.tensor
		average softmax matrix of the visual similarities between each supergroup

	Return
	------
	ordered super_group_list: list
		list of supergroups where each supergroup is a list containing similar groupIDs
	"""

	similar_groups = create_supergroup_list_helper(number_of_classes, grouping_volatility, avg_softmax_matrix)
	similar_groups_dict = transitive_grouper(similar_groups)

	seen = []
	super_group_list = [] # List of groups where the groups are identified by their groupID

	for idx in similar_groups_dict:
		if(idx not in seen):
			super_group_list.append(similar_groups_dict[idx])
			for jdx in similar_groups_dict[idx]:
				seen.append(jdx)

	return [sorted(group) for group in super_group_list]

def save_supergroup_list(dataloader, current_supergroup, super_group_list):
	"""
	save the list of supergroups to a json file

	Parameters
	----------
	dataloader: torch.utils.data.DataLoader
		the iterable dataloader we are using

	current_supergroup: str
		the current supergroup we are examining

	super_group_list: list
		list of supergroups where each supergroup is a list containing similar groupIDs
	"""

	path = os.path.join(dataloader.dataset.path_to_outputs, f"{current_supergroup}_supergroups.json")
	super_group_dict = {f"Group {idx + 1}": super_group_list[idx] for idx in range(len(super_group_list))}

	with open(path, "w", encoding='utf-8') as fptr:
		fptr.write(json.dumps(super_group_dict, indent=4))

def ModelVisualSimilarityMetric(dataloader, current_supergroup, grouping_volatility, path_to_softmax_matrix, debug=False):
	"""
	Create a list of supergroups based on the softmax matrix for the current supergroup

	dataloader: torch.utils.data.DataLoader
		the iterable dataloader we are using

	current_supergroup: str
		the current supergroup we are examining

	grouping_volatility: float
		constant for sigmoidal membership

	path_softmax_matrix: str
		path to the current supergroup's softmax matrix

	debug: bool
		debug flag to print outputs

	Return
	------
	super_group_list: list
		list of supergroups where each supergroup is a list containing similar groupIDs
	"""

	avg_softmax_matrix = torch.load(path_to_softmax_matrix, map_location=device)
	number_of_classes = avg_softmax_matrix.shape[0]
	if(debug):
		print(f"Number of classes: {number_of_classes}")
	
	super_group_list = create_supergroup_list(number_of_classes, grouping_volatility, avg_softmax_matrix)
	if(debug):
		print(f"Number of Groups: {len(super_group_list)}")
		for idx, group in enumerate(super_group_list):
			print(f"Group {idx + 1}: {group}")
	
	save_supergroup_list(dataloader, current_supergroup, super_group_list)
	return super_group_list

def update_target_map(dataloader, current_supergroup, grouping_volatility, path_to_softmax_matrix, supergroup_path, debug=False):
	"""
	update the target_map to add new supergroups if necessary based on the softmax matrix of the current trained supergroup

	Parameter
	---------
	dataloader: torch.utils.data.DataLoader
		iterable training dataloader

	current_supergroup: str
		the current supergroup we are examining

	grouping_volatility: float
		constant for sigmoidal membership

	path_to_softmax_matrix: str
		path to the current supergroup's softmax matrix

	supergroup_path: list
		path to supergroup as given by path_decisions

	debug: bool
		debug flag to print outputs

	Return
	------
	list_of_new_supergroups: list
		the list of supergroups that are present in the updated_path_decisions but not in the previous_path_decisions. These are new modules
	"""

	def find_categories_beloning_to_sg(target_map_):
		"""
		find all the groups or categories that belong to the current supergroup

		Parameters
		----------
		target_map_: dict
			the previous target_map

		Return
		------
		classes_belonging_to_current_supergroup: dict
			all the classes and their respective paths down the tree that belong to the current supergroup
		"""

		classes_belonging_to_current_supergroup = {} # dictionary of classes and their respective paths that belong to the current supergroup
		if len(supergroup_path) == 0:
			# if len(supergroup_path == 0) then we're at the root node so the classes_belonging_to_current_supergroup = target_map as they all 
			# belong to root
			classes_belonging_to_current_supergroup = target_map_

		else:
			for category_encoding, path_to_sg in target_map_.items():
				for idx in range(len(supergroup_path)):
					if(path_to_sg[idx] != supergroup_path[idx]):
						# if the path down the tree for this current category does not match the path down to the supergroup we are examining then 
						# this category does not belong to the supergroup so we break and move to the next category
						break

					elif(idx == len(supergroup_path) - 1):
						# the entire path has matched up so this category definitely belongs to this supergroup
						classes_belonging_to_current_supergroup[category_encoding] = path_to_sg

		return classes_belonging_to_current_supergroup	
	
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
	
	def new_supergroups():
		"""
		Identify the supergroups that are present in the updated_path_decisions but not in the previous_path_decisions

		Return
		------
		new_groups: list
			the list of supergroups that are present in the updated_path_decisions but not in the previous_path_decisions. These are new modules
		"""

		previous = set(previous_path_decisions.keys())
		updated = set(updated_path_decisions.keys())
		return list(previous.symmetric_difference(updated))

	super_group_list = ModelVisualSimilarityMetric(dataloader, current_supergroup, grouping_volatility, path_to_softmax_matrix, debug) 
	previous_nodes_dict = dataloader.dataset.get_dictionary_of_nodes()
	if(len(super_group_list) == 1):
		# Don't create any new children if there are no sub-groups to be made
		print(f"No new children made for supergroup {current_supergroup}")
		previous_nodes_dict[current_supergroup].num_groups = len(previous_nodes_dict[current_supergroup].classes)
		dataloader.dataset.update_tree_attributes(previous_nodes_dict) # update these attributes in the tree
		return []
	
	previous_target_map = dataloader.dataset.get_target_map()
	previous_path_decisions = dataloader.dataset.get_path_decisions()
	
	updated_target_map = copy.deepcopy(previous_target_map) 
	categories_that_belong_to_super_group = find_categories_beloning_to_sg(updated_target_map) 
	checked_categories = []
	supergroup_id = 0 # the id of the next child or supergroup node child of the current supergroup we are examining
	
	# iterate through the supergroup_list and examine each new supergroup
	for supergroup in super_group_list:
		id_within_supergroup = 0 # id of the leaf node within the new supergroup
		for category_encoding, path_to_sg in categories_that_belong_to_super_group.items():
			path_to_sg = remove_padding_from_path(path_to_sg)
			if(category_encoding in checked_categories):
				pass

			elif(path_to_sg[-1] in supergroup): # if the category is in this new supergroup
				if(len(supergroup) == 1): # if there is only one element in this supergroup, then assign it the supergroup_id
					path_to_sg[-1] = supergroup_id

				else:
					path_to_sg.append(id_within_supergroup) # add the new leaf index within the new supergroup
					path_to_sg[-2] = supergroup_id # change the second to last index in the path to accomodate for the new supergroup introduced
					id_within_supergroup += 1 # increment the id_within_supergroup to accomodate for another category that may belong to this new supergroup

				checked_categories.append(category_encoding) 
				updated_target_map[category_encoding] = path_to_sg 

		supergroup_id += 1
	
	dataloader.dataset.update_target_map(updated_target_map)
	dataloader.dataset.update_tree(current_supergroup) 
	dataloader.dataset.padding_target_map()
	updated_path_decisions = dataloader.dataset.get_path_decisions()

	list_of_new_supergroups = new_supergroups()
	for new_sg in list_of_new_supergroups:
		# Iterate through this list and remove supergroups that belong to the previous paths and not the updated one
		# This ensures that a label that has been grouped into a different node in the updated version is the only one remaining 
		if(new_sg not in updated_path_decisions and new_sg in previous_path_decisions and previous_nodes_dict[new_sg].is_trained == False):
			list_of_new_supergroups.remove(new_sg)

	return list_of_new_supergroups