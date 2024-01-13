# ----------------------------------------------------
# Name: VGGNet_Backbone.py
# Purpose: Script to create the vgg class for svhn dataset
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torch
import torch.nn as nn

class MNN(nn.Module):
	def __init__(self, supergroup, number_of_classes, input_shape, debug_flag=True) -> None:
		super(MNN, self).__init__()
		self.number_of_classes = number_of_classes
		self.features = self._make_layer(supergroup, input_shape[0])
		self.debug_flag = debug_flag

		self.sample_input = torch.unsqueeze(torch.ones(input_shape), dim=0) # input shape = 1 x channels x height x width with ones as dummy input
		if(self.debug_flag):
			print(f"vggMNN: sample_input.shape = {self.sample_input.shape}")
		
		self.classifier = nn.Identity() # temporarily create a classifier that does nothing to the input so we can determine the shape of the feature_map
		feature_map, classifier_features = self.forward(self.sample_input) #TODO: check this self() == self.forward()?
		if(self.debug_flag):
			print(f"vggMNN: feature_map.shape = {feature_map.shape}")
			print(f"vggMNN: classifier_features.shape = {classifier_features.shape}")

		self.classifier = nn.Sequential(
			nn.Linear(in_features=classifier_features.shape[1], out_features=self.number_of_classes)
		)
		
	def _make_layer(self, supergroup, input_channel):
		layers = []
		if(supergroup == "root"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, stride=1, padding=1))

			layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU6(inplace=True))

			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU6(inplace=True))

			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU6(inplace=True))

			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			layers.append(nn.Dropout2d(0.2))

		else:
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, stride=1, padding=1))
			layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			layers.append(nn.Dropout2d(0.2))

		return nn.Sequential(*layers)
	
	def forward(self, x):
		features = self.features(x)
		features = features.view(features.shape[0], -1)
		prediction = self.classifier(features)
		return features, prediction 
	
	def evaluate(self, x):
		self.eval()
		return self.forward(x)