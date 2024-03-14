# ----------------------------------------------------
# Name: pathDecisions.py
# Purpose: Script to build the TRUNK tree
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import re
import pickle
import json
from collections import deque

class TreeNode:
    """
    A class used to represent a node in the tree 

    Attributes
    ----------
    value: str
        The supergroup notation (i.e. sg1, sg2, root) for the node

    parent: TreeNode
        the parent of the current node

    classes: list
        list of dataset categories that belong to this node

    children: list
        the list of children this node is a parent to

    num_groups: int
        number of children this node has

    output_image_shape: tuple
        the output image shape that comes out of this node

    is_trained: bool
        flag to let us know if the node has been trained or not
    """

    def __init__(self, value, parent, classes=[], num_groups=0):
        """
        Parameters
        ----------
        value: str
            The supergroup notation (i.e. sg1, sg2, root) for the node  

        parent: TreeNode
            the parent node for this new node

        classes: list
            list of classes from the dataset that correspond to this new node     
        """

        self.value = value 
        self.classes = classes 
        self.parent = parent 
        self.children = [] 
        self.num_groups = num_groups 
        self.output_image_shape = () 
        self.is_trained = False  

    def depth_of_tree(self):
        """
        Compute the depth of the tree

        Return
        ------
        depth: int
            depth of the tree
        """

        if(not self.children):
            return 1
        
        return 1 + max(child.depth_of_tree() for child in self.children)
    
def get_group_number(group_name):
    """
    Given a node/group name sg5 for example, extract the interger 5 from the name

    Parameters
    ----------
    group_name: str
        current node name or value

    Return
    ------
    group_number: int
       the integer extracted from the group name 
    """

    pattern = r"(sg)([0-9]+)" # Regex pattern to extract the integer in the second group from the string
    return int(re.sub(pattern, r"\2", group_name))

def create_value_for_node(nodes_dict, node):
    """
    Create a group name or value for the current node based on the previous node values already on the tree

    Parameters
    ----------
    nodes_dict: dict
        dictionary of nodes on the tree where the (key, value) pair of the dictionary is (value of node, TreeNode)

    node: TreeNode
        the parent node of the current node we're looking at in the build_tree_from_input function
    """

    value = "sg" + str(len(node.children) + 1) # Let the new child's value be one larger than the maximum supergroup number in the list of children (i.e. sg4 if the node's current children is [sg1, sg2, sg3])
    if(value not in nodes_dict): 
        # If this value doesn't already exist in the dictionary of all the nodes of the tree, then return this value for the new child node
        return value

    group_numbers = [int(get_group_number(group_name)) for group_name in list(nodes_dict.keys()) if group_name != "root"]
    return "sg" + str(int(max(group_numbers) + 1))

def build_tree_from_input(target_map, nodes_dict):
    """
    Given the target_map, build a tree and add new nodes if necessary 

    Parameters
    ----------
    target_map: dict
        dictionary mapping of the paths down the tree given the categoryId
    
    nodes_dict: dict 
        dictionary of nodes on the tree where the (key, value) pair of the dictionary is (value of node, TreeNode)

    Return
    ------
    nodes_dict: dict
        updated dictionary of nodes with new nodes added to it
    """
    
    for value, node in nodes_dict.items():
        if(value != "root"):
            node.classes = []

    root = nodes_dict['root']

    # Iterate through the target_map to build the tree
    for label, path in target_map.items():
        node = root 
        for idx in path: 
            if(idx == -1):
                # -1 is used for padding so we want to omit this in the tree build
                continue
            else:
                if idx >= len(node.children): 
                    child_value = create_value_for_node(nodes_dict, node) 
                    child = TreeNode(value=child_value, parent=node, classes=[]) 
                    node.children.append(child) 
                    node.num_groups += 1
                else:
                    child = node.children[idx] 

                if(label not in child.classes):
                    child.classes.append(label) 

                node = child 
                nodes_dict[child.value] = child 

    return nodes_dict

def build_path_decisions(nodes_dict):
    """
    Given the dictionary of nodes, now create a dictionary mapping of each supergroup/node and the path down to it from the root node

    Parameters
    ----------
    nodes_dict: dict
        dictionary of nodes on the tree where the (key, value) pair of the dictionary is (value of node, TreeNode)

    Return
    ------
    path_decisions: dict
        dictionary mapping of every supergroup/node and the path down to it from the root nodes
    """

    path_decisions = {} 

    def dfs(node, path_indices):
        """
        Use depth-first search to recursively iterate through the tree and add the path down to the supergroup/node from the root node
        
        Parameters
        ----------
        node: TreeNode
            the current node we are examining in the tree

        path_indices: list
            list of indexed nodes in the path

        Return
        ------
        """

        path_decisions[node.value] = path_indices 
        for idx, child in enumerate(node.children):
            dfs(child, path_indices + [idx])

    dfs(nodes_dict['root'], [])
    return path_decisions

def build_path_values(nodes_dict):
    """
    Given the dictionary of nodes, now create a dictionary mapping of each supergroup/node and the path down to it from the root node by its value rather than indices

    Parameters
    ----------
    nodes_dict: dict
        dictionary of nodes on the tree where the (key, value) pair of the dictionary is (value of node, TreeNode)

    Return
    ------
    path_decisions: dict
        dictionary mapping of every supergroup/node and the path down to it from the root nodes by their values (i.e. SG1: [root, SG1])
    """

    path_values = {}

    def dfs(node, path):
        """
        Use depth-first search to recursively iterate through the tree and add the path down to the supergroup/node from the root node
        
        Parameters
        ----------
        node: TreeNode
            the current node we are examining in the tree

        path: list
            list of valued nodes in the path

        Return
        ------
        """

        path_values[node.value] = path[:]
        for child in node.children:
            dfs(child, path + [child.value])

    dfs(nodes_dict['root'], ['root'])
    return path_values

def write_tree_to_file(root, path):
    """
    Write the current tree to a pickle file

    Parameters
    ----------
    root: TreeNode
        the root of the tree

    path: str
        the path to the file in which we want to save the current tree
    """

    with open(path, 'wb') as f:
        pickle.dump(root, f)

def load_tree_from_file(path):
    """
    Read the pickle file and load the tree

    Parameters
    ----------
    path: str
        the path to the pickle file storing the tree

    Return
    ------
    nodes_dict: dict
        dictionary of all the nodes in the tree
    """

    with open(path, 'rb') as fptr:
        root = pickle.load(fptr) 

    def build_nodes_dict_from_saved_tree(node, nodes_dict):
        """
        recursively create the dictionary of nodes where the (key, value) pair is (value of the node, TreeNode) from the contents of the pickle file

        Parameters
        ----------
        node: TreeNode
            the current node we're examining

        nodes_dict: dict
            dictionary of all the nodes in the tree

        Return
        ------
        nodes_dict: dict
            dictionary of all the nodes in the tree
        """

        if node is None:
            return
        
        nodes_dict[node.value] = node
        for child in node.children:
            build_nodes_dict_from_saved_tree(child, nodes_dict)
        
        return nodes_dict

    return build_nodes_dict_from_saved_tree(root, {})

def get_path_decisions(path_to_tree):
    """
    Given the target_map, this function will return a dictionary that maps the path down to the supergroups from the root node based on their index positions

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    Return
    ------
    path_decisions: dict
        dictionary that maps the path down to the supergroups from the root node where the list is the index positions in the tree
    """

    nodes = load_tree_from_file(path_to_tree) 
    path_decisions = build_path_decisions(nodes)

    return path_decisions

def get_leaf_nodes(path_to_tree):
    """
    Get a dictionary of nodes that are the leaves of the tree and its respective paths from the root

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    Return
    ------
    leafs: dict
        dictionary of leaf nodes and its respective path from the root node using path decisions
    """

    nodes = load_tree_from_file(path_to_tree) 
    path_decisions = get_path_decisions(path_to_tree)

    leafs = {}
    for sg_name, sg_node in nodes.items():
        if(len(sg_node.classes) == 1):
            leafs[sg_name] = path_decisions[sg_name]

    return leafs

def get_paths_to_leaf(path_to_tree):
    """
    Get the path down to the leaf nodes from root using paths

    Parameter
    ---------
    path_to_tree: str
        the path to the file storing the tree information

    Return
    ------
    leafs: dict
        dictionary of leaf nodes named as their labels and its respective path from the root node using paths
    """

    nodes = load_tree_from_file(path_to_tree)
    paths = get_paths(path_to_tree)

    leafs = {}
    for sg_name, sg_node in nodes.items():
        if(len(sg_node.classes) == 1):
            leafs[sg_node.classes[0]] = paths[sg_name]

    return leafs

def map_leaf_name_to_category(path_to_tree):
    """
    Get the path down to the leaf nodes from root using paths

    Parameter
    ---------
    path_to_tree: str
        the path to the file storing the tree information

    Return
    ------
    leafs: dict
        dictionary of leaf nodes mapped with the class they are responsible for
    """

    nodes = load_tree_from_file(path_to_tree)
    leafs = {}
    for sg_name, sg_node in nodes.items():
        if(len(sg_node.classes) == 1):
            leafs[sg_name] = sg_node.classes[0]

    return leafs

def update_number_groups(current_supergroup, path_to_weights, nodes):
    """
    Update the number of groups the supergroup has after training and grouping

    Parameters
    ----------
    current_supergroup: str
        the current supergroup we are examining

    path_to_weights: str
        the path to the json file containing the different groups pertaining to the current supergroup

    nodes: dict
        dictionary of nodes

    Return
    ------
    nodes: dict
        updated dictionary of nodes
    """

    fptr = open(path_to_weights, "r")
    groups = json.load(fptr)
    nodes[current_supergroup].num_groups = len(groups)
    return nodes

def get_paths(path_to_tree):
    """
    Given the target_map, this function will return a dictionary that maps the path down to the supergroups from the root node based on the node values

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    Return
    ------
    paths: dict
        dictionary that maps the path down to the supergroups from the root node where the list are the node values
    """

    nodes = load_tree_from_file(path_to_tree) 
    paths = build_path_values(nodes) 

    return paths

def update_tree(path_to_tree, target_map, current_supergroup=None, path_to_weights=None):
    """
    Update the tree based on the new target_map

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    current_supergroup: str [Optional]
        the current supergroup we're at that we want to indicate is trained

    target_map: dict
        dictionary mapping the path for each category from the root node

    path_to_weights: str [Optional]
        the path to the json file containing the different groups pertaining to the current supergroup
    """
    
    if(target_map == {str(idx): [idx] for idx in range(len(target_map))}): # if the target_map is in its starting phase (i.e. {0: [0], 1:[1], 2: [2], ...}), initialize a path_decisions variable
        nodes = {'root': TreeNode(value='root', parent=None, classes=list(target_map.keys()), num_groups=len(target_map))} # create the root node and add it to the dictionary of nodes (nodes_dict)

    else:
        nodes = load_tree_from_file(path_to_tree) 
        nodes = build_tree_from_input(target_map, nodes) 
        nodes = remove_nodes_with_empty_classes(nodes) # some nodes will end up with no class so we will remove this

        if(path_to_weights):
            # update the number of supergroup children the nodes has after grouping
            nodes = update_number_groups(current_supergroup, path_to_weights, nodes)

    write_tree_to_file(nodes['root'], path_to_tree) # write the new tree to the originally saved file (path from root but the list is the value of nodes)

def update_tree_attributes(path_to_tree, nodes):
    """
    Update the attributes of the tree

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    nodes: dict
        dictionary of nodes in the tree
    """

    write_tree_to_file(nodes["root"], path_to_tree)

def remove_nodes_with_empty_classes(nodes_dict):
    """
    Some nodes will end up with no class so we will remove this from the tree

    Parameters
    ----------
    nodes_dict: dict
        dictionary of the nodes in the tree
    
    Return
    ------
    nodes_dict: dict
        updated dictionary of nodes
    """

    remove_nodes = []
    for value, node in nodes_dict.items():
        if(len(node.classes) == 0):
            parent = node.parent
            parent.children.remove(node)
            remove_nodes.append(value)

    for node_value in remove_nodes:
        del nodes_dict[node_value]

    return nodes_dict

def get_num_classes_per_sg(path_to_tree):
    """
    Get a dictionary mapping each node and the number of classes they are responsible for

    Parameters
    ----------
    path_to_tree: str
        path to the file storing the tree information

    Return
    ------
    num_class_per_sg: dict
        dictionary mapping each node and the number of classes they are responsible for
    """

    nodes = load_tree_from_file(path_to_tree) 
    num_class_per_sg = {}

    for node_value, node in nodes.items():
        num_class_per_sg[node_value] = len(node.classes)

    return num_class_per_sg

if __name__ == "__main__":
    grouping_volatility = str(input("Enter grouping volatility: "))
    model_backbone = input("Enter model backbone: ")
    path = f"/home/ravi30/TRUNK_Tutorial_Paper/TRUNK/Datasets/cifar10/{model_backbone}/{grouping_volatility}/tree.pkl"
    tree = load_tree_from_file(path)

    sgs = {}
    for value, node in tree.items():
        if(len(node.classes) > 1):
            sgs[value] = node

    for value, node in sgs.items():
        if(value != "root"):
            print(f"For node {value} its children are {[child.value for child in node.children]}, its classes are {node.classes}, and its parent is {node.parent.value}")
        else:
            print(f"For node {value} its children are {[child.value for child in node.children]}, its classes are {node.classes}")
