# ----------------------------------------------------
# Name: MobileNet_Backbone.py
# Purpose: Script to create the mobilenet class for cifar dataset
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary packages
import torch
import torch.nn as nn

### MobileNet Architecture
class DepthwiseConvolutionBlock(nn.Module):
	def __init__(self, in_ch, kernel_size, padding=1, bias=False, stride=1):
		super(DepthwiseConvolutionBlock, self).__init__()
		self.depthwise_convolution_block = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias),
			nn.BatchNorm2d(in_ch, eps=1e-5, momentum=0.1, affine=True),
			nn.ReLU6(True)
		)
	def forward(self, x):
		out = self.depthwise_convolution_block(x)
		return out

class Conv1x1Block(nn.Module):
	def __init__(self, ch_in, ch_out, padding=1, bias=False, stride=1):
		super(Conv1x1Block, self).__init__()
		self.conv_1x1_block = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=padding, bias=bias),
			nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True),
			nn.ReLU6(True)
		)
	def forward(self, x):
		out = self.conv_1x1_block(x)
		return out
	
class InvertedResidual(nn.Module):
	def __init__(self, ch_in, ch_out, width_multiplier=1, stride=1):
		super(InvertedResidual, self).__init__()
		self.hidden_dimension = width_multiplier * ch_in
		
		layers = []
		if(width_multiplier != 1):
			layers.append(Conv1x1Block(ch_in=ch_in, ch_out=self.hidden_dimension))
		layers.extend([DepthwiseConvolutionBlock(in_ch=self.hidden_dimension, kernel_size=3, stride=stride),
				 Conv1x1Block(ch_in=self.hidden_dimension, ch_out=ch_out)])

		self.inverted_residual = nn.Sequential(*layers)

	def forward(self, x):
		feature_map = self.inverted_residual(x)
		if(feature_map.shape[1:] == x.shape[1:]):
			feature_map += x # Add input to output if shape is the same
		
		return feature_map

# MobileNet layers will be arranged based on the supergroup configuration in MNN
class MNN(nn.Module):
	def __init__(self, supergroup, number_of_classes, input_shape, debug_flag=True):
		super(MNN, self).__init__()
		self.number_of_classes = number_of_classes
		self.debug_flag = debug_flag
		self.features = self._make_layer(supergroup, input_shape[0]) # Feature extraction of the image passed through based on the supergroup
		self.sample_input = torch.unsqueeze(torch.ones(input_shape), dim=0) # input shape = 1 x channels x height x width with ones as dummy input
		if(self.debug_flag):
			print(f"MobileNetMNN: sample_input.shape = {self.sample_input.shape}")
		
		self.classifier = nn.Identity() # temporarily create a classifier that does nothing to the input so we can determine the shape of the feature_map
		feature_map, _ = self.forward(self.sample_input) 
		if(self.debug_flag):
			print(f"MobileNetMNN: feature_map.shape = {feature_map.shape}")

		self.classifier = nn.Sequential(
			nn.Conv2d(feature_map.shape[1], self.number_of_classes, kernel_size=1,stride=1,padding=0),
			nn.AvgPool2d((feature_map.shape[2],feature_map.shape[3])),
			nn.LogSoftmax(dim=1),
			nn.Flatten()
		)

	def _make_layer(self, supergroup, input_channel):
		layers = []
		if(supergroup == "root"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=32))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(InvertedResidual(ch_in=32, ch_out=64, width_multiplier=2, stride=1))
			layers.append(InvertedResidual(ch_in=64, ch_out=64, width_multiplier=4, stride=1))

		elif(supergroup == "sg5"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=128))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(InvertedResidual(ch_in=64, ch_out=128, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=128, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=256, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=256, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=256, width_multiplier=2, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=512, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=512, ch_out=512, width_multiplier=2, stride=1))
			###
			layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=1024))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1))
			layers.append(nn.BatchNorm2d(num_features=512))
			layers.append(nn.ReLU6(inplace=True))

		elif(supergroup == "sg15"):
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=256))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=128))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(InvertedResidual(ch_in=64, ch_out=128, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=128, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=256, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=256, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=256, width_multiplier=2, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=512, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=512, ch_out=512, width_multiplier=2, stride=1))
			###
			layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=1024))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1))
			layers.append(nn.BatchNorm2d(num_features=512))
			layers.append(nn.ReLU6(inplace=True))

		else: # Every other supergroup
			layers.append(nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(num_features=64))
			layers.append(nn.ReLU6(inplace=True))
			###
			layers.append(InvertedResidual(ch_in=64, ch_out=64, width_multiplier=2, stride=1))
			layers.append(InvertedResidual(ch_in=64, ch_out=128, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=128, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=264, width_multiplier=16, stride=1))
			###
			layers.append(InvertedResidual(ch_in=264, ch_out=512, width_multiplier=2, stride=1))
			layers.append(InvertedResidual(ch_in=512, ch_out=256, width_multiplier=4, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=128, width_multiplier=8, stride=1))
			layers.append(InvertedResidual(ch_in=128, ch_out=256, width_multiplier=16, stride=1))
			layers.append(InvertedResidual(ch_in=256, ch_out=256, width_multiplier=8, stride=1))

		return nn.Sequential(*layers)
	
	def forward(self, x):
		features = self.features(x)
		prediction = self.classifier(features)
		return features, prediction
	
	def evaluate(self, x):
		self.eval()
		return self.forward(x)