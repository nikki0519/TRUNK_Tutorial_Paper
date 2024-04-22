# Reproducible Deep Learning Software for Efficient Computer Vision 
PyTorch implementation and pre-trained models of TRUNK for the EMNIST, CiFAR10, and SVHN datasets. For details, see the papers: [Reproducible Deep Learning Software for Efficient Computer Vision](#Include Link).

Computer vision (CV) using deep learning can equip machines with the ability to understand visual information. CV has
seen widespread adoption across numerous industries, from autonomous vehicles to facial recognition on smartphones.
However, alongside these advancements, there have been increasing concerns about reproducing the results. The difficulty of
reproducibility may arise due to multiple reasons, such as differences in execution environments, missing or incompatible
software libraries, proprietary data, and the stochastic nature in some software. A study conducted by the Nature journal
reveals that more than 70% of researchers failed to reproduce other researcher’s experiments; over 50% failed to reproduce
their own experiments. Given the critical role that computer vision plays in many applications, especially in edge devices
like mobile phones and drones, irreproducibility poses significant challenges for researchers and practitioners. To address
these concerns, this paper presents a systematic approach at analyzing and improving the reproducibility of computer vision
models through case studies. This approach combines rigorous documentation standards, standardized software environment,
and a comprehensive guide of best practices. By implementing these strategies, we aim to bridge the gap between research
and practice, ensuring that innovations in computer vision can be effectively reproduced and deployed.

This github repository contains all the code used to generate the results mentioned or displayed in the aforementioned paper, divided into three directories:
- LPCV Background: Contains the code used to introduce LPCV techniques mentioned in the background section of the paper
- Monolithic Architectures: Contains the code used to compare the TRUNK architecture against well-known monolithic architectures such as ConvNeXt, DinoV2, MobileNetv2, ResNet, ResNet Quantized, VGG Pruned, VGG, and ViT
- TRUNK: Contains the code for training and testing the TRUNK network, as well as the pre-trained models

## Installation
The implementations provided in this repository requires a python version 3.9.18, a PyTorch version 2.1 with CUDA 11.8, and other 3rd party packages. To setup the required dependencies to reproduce the results of this repository, follow these instructions

[pip][2] (**Recommended**) - clone the repository and then install the dependencies using the provided command
```bash
pip install -r requirements.txt
```

[conda][1] - clone the repository and then activate the mnn conda environment using the provided environment definition
```bash
conda env create -f environment.yml
conda activate mnn
```

[1]: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
[2]: https://pip.pypa.io/en/stable/getting-started/