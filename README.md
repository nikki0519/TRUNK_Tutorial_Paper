# TRUNK Tutorial Paper
PyTorch implementation and pre-trained models of TRUNK for the EMNIST, CiFAR10, and SVHN datasets. For details, see the papers: [A Tutorial Paper for Tree-based Unidirectional Neural Networks for Low Powered Computer Vision (LPCV)](#Include Link).

Despite effectively reducing the memory requirements, most low-powered computer vision techniques still retain their monolithic structures that rely on a single DNN to simultaneously classify all the categories, causing it to still conduct redundant operations. To address this issue, we introduce a tree-based unidirectional neural network (TRUNK) to eliminate redundancies for image classification. The TRUNK architecture reduces these redundant operations by using only a small subsets of the neurons in the DNN by using a tree-based approach. 

This github repository contains all the code used to generate the results mentioned or displayed in the aforementioned paper, divided into three directories:
- LPCV Background: Contains the code used to introduce LPCV techniques mentioned in the background section of the paper
- Monolithic Architectures: Contains the code used to compare our TRUNK architecture against well-known monolithic architectures such as ConvNeXt, DinoV2, MobileNetv2, ResNet, ResNet Quantized, ResNet Pruned, VGG, and ViT
- TRUNK: Contains the code for training and testing our TRUNK, as well as the pre-trained models

## Installation
The implementations provided in this repository requires a python version 3.9.18, a PyTorch version 1.12 with CUDA 11.3, and other 3rd party packages. To setup the required dependencies to reproduce the results of this repository, follow these instructions

[conda][1] (**Recommended**) - clone the repository and then activate the mnn conda environment using the provided environment definition
```bash
conda env create -f environment.yml
conda activate mnn
```

[pip][2] - clone the repository and then install the dependencies using the provided command
```bash
pip install -r requirements.txt
```

[1]: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
[2]: https://pip.pypa.io/en/stable/getting-started/