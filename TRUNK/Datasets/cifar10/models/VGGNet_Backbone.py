# ----------------------------------------------------
# Name: VGGNet_Backbone.py
# Purpose: Script to create the vgg class for cifar dataset
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torch
import torch.nn as nn

class DynamicNormalization(nn.Module):
	def __init__(self, batch_norm=True):
		super(DynamicNormalization, self).__init__()
		self.batch_norm = batch_norm

	def forward(self, x):
		N, C, H, W = x.shape
		if(self.batch_norm):
			norm_layer = nn.BatchNorm2d(num_features=C).to(x.device)
		else:
			norm_layer = nn.LayerNorm([C, H, W]).to(x.device)

		return norm_layer(x)

class MNN(nn.Module):
	def __init__(self, supergroup, number_of_classes, input_shape, debug_flag=True) -> None:
		super(MNN, self).__init__()
		self.supergroup = supergroup
		self.number_of_classes = number_of_classes
		self.features = self._make_layer(input_shape[0], batch_norm=True)
		self.debug_flag = debug_flag

		self.sample_input = torch.unsqueeze(torch.ones(input_shape), dim=0) # input shape = 1 x channels x height x width with ones as dummy input
		if(self.debug_flag):
			print(f"vggMNN: sample_input.shape = {self.sample_input.shape}")
		
		self.classifier = nn.Identity() # temporarily create a classifier that does nothing to the input so we can determine the shape of the feature_map
		feature_map, classifier_features = self.forward(self.sample_input)
		if(self.debug_flag):
			print(f"vggMNN: feature_map.shape = {feature_map.shape}")
			print(f"vggMNN: classifier_features.shape = {classifier_features.shape}")

		self.classifier = nn.Sequential(
			nn.Linear(in_features=classifier_features.shape[1], out_features=self.number_of_classes)
		)
		
	def _make_layer(self, input_channel, batch_norm=True):
		layers = []
		if(self.supergroup == "root"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=16)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			###
			layers.append(nn.Dropout2d(p=0.5))

		elif(self.supergroup == "sg1" or self.supergroup == "sg3"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Dropout2d(p=0.5))

		elif(self.supergroup == "sg5"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64))
			layers.append(nn.ReLU(inplace=True))
			###
			# Comment this part
			layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=128))
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=128))
			layers.append(nn.ReLU(inplace=True))
			# To this part
			###
			layers.append(nn.Dropout2d(p=0.5))
			
		else:
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=16)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=128)) 
			layers.append(nn.ReLU(inplace=True))
			###
			layers.append(nn.Dropout2d(p=0.5))

		layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
		return nn.Sequential(*layers)
	
	def forward(self, x):
		features = self.features(x)
		features_flattend = features.view(features.shape[0], -1)
		prediction = self.classifier(features_flattend)
		return features, prediction

	def evaluate(self, x):
		self.eval()
		return self.forward(x)