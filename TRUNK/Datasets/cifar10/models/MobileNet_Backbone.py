# ----------------------------------------------------
# Name: MobileNet_Backbone.py
# Purpose: Script to create the mobilenet class for cifar dataset
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspired by https://github.com/kuangliu/pytorch-cifar

class Block(nn.Module):
	def __init__(self, ch_in, ch_out, stride=1):
		super(Block, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3, stride=stride, padding=1, groups=ch_in, bias=False)
		self.bn1 = nn.BatchNorm2d(num_features=ch_in)
		self.conv2 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(num_features=ch_out)
		
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		return out

# MobileNet layers will be arranged based on the supergroup configuration in MNN
class MNN(nn.Module):
	def __init__(self, supergroup, number_of_classes, input_shape, debug_flag=True):
		super(MNN, self).__init__()
		self.supergroup = supergroup
		self.input_shape = input_shape
		self.number_of_classes = number_of_classes
		self.debug_flag = debug_flag
		self.features = self._make_layer(self.input_shape[0]) # Feature extraction of the image passed through based on the supergroup
		
		self.sample_input = torch.unsqueeze(torch.ones(self.input_shape), dim=0) # input shape = 1 x channels x height x width with ones as dummy input
		if(self.debug_flag):
			print(f"MobileNetMNN: sample_input.shape = {self.sample_input.shape}")
		
		self.classifier = nn.Identity() # temporarily create a classifier that does nothing to the input so we can determine the shape of the feature_map
		feature_map, classifier_features = self.forward(self.sample_input) 
		if(self.debug_flag):
			print(f"MobileNetMNN: feature_map.shape = {feature_map.shape}")
			print(f"MobileNetMNN: classifier_features.shape = {classifier_features.shape}")

		self.classifier = nn.Sequential(
			nn.Linear(in_features=classifier_features.shape[1], out_features=self.number_of_classes)
		)

	def _make_layer(self, input_channel):
		layers = []
		if(self.supergroup == "root"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(Block(ch_in=32, ch_out=64, stride=1))
			layers.append(Block(ch_in=64, ch_out=128, stride=2))
			layers.append(Block(ch_in=128, ch_out=128, stride=1))
			layers.append(Block(ch_in=128, ch_out=256, stride=2))
			layers.append(Block(ch_in=256, ch_out=256, stride=1))
			layers.append(Block(ch_in=256, ch_out=512, stride=2))
			layers.append(Block(ch_in=512, ch_out=512, stride=1))
			layers.append(Block(ch_in=512, ch_out=512, stride=1))

		else: # Every other supergroup
			layers.append(Block(ch_in=input_channel, ch_out=512, stride=1))
			layers.append(Block(ch_in=512, ch_out=512, stride=1))
			layers.append(Block(ch_in=512, ch_out=512, stride=1))
			layers.append(Block(ch_in=512, ch_out=1024, stride=2))
			layers.append(Block(ch_in=1024, ch_out=1024, stride=1))

		return nn.Sequential(*layers)
	
	def forward(self, x):
		features = self.features(x)
		features = F.avg_pool2d(features, kernel_size=2)
		features_flattend = features.view(features.shape[0], -1)
		prediction = self.classifier(features_flattend)
		return features, prediction
	
	def evaluate(self, x):
		self.eval()
		return self.forward(x)