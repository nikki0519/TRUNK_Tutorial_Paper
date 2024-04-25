# TRUNK
Official PyTorch implementation and pre-trained models of TRUNK for the EMNIST, CiFAR10, and SVHN datasets. For details, see the papers: [Reproducible Deep Learning Software for Efficient Computer Vision](#Include Link).

Despite effectively reducing the memory requirements, most low-powered computer vision techniques still retain their monolithic structures that rely on a single DNN to simultaneously classify all the categories, causing it to still conduct redundant operations. To address this issue, the Tree-based Unidirectional Neural Network (TRUNK) was introduced to eliminate redundancies for image classification. The TRUNK architecture reduces these redundant operations by using only a small subsets of the neurons in the DNN by using a tree-based approach. 

The Datasets directory is further divided into the three folderes representing the three datasets (i.e. EMNSIT, CiFAR10, SVHN) we've used to train/test TRUNK. Within these folders, we have the models/ directory where the MobileNetv2 and VGG16 networks are saved. Within the mobilenet/ and vgg/ directories, and the respective grouping volatility folders, the inputs and results (i.e. hyper-parameters, model weights, model softmax, details of the tree, and inference results) are stored. 

The architecture design varies by the different datasets used and is inspired by the [MobileNetv2][1] and [VGG-16][2] networks. The pre-trained weights for EMNIST and SVHN are available for the MobileNet inspired architecture. The pre-trained weights for CIFAR-10 are available for the VGG inspired network. The networks for each dataset are found in Datasets/dataset name/models

The data (EMNIST, CiFAR10, and SVHN) used to train and test the TRUNK model will be downloaded from torchvision when executing the main.py script as demonstrated in the training section. 
## Training
To train TRUNK in the paper for a particular dataset, run this command:

```train
$ python main.py --train --dataset emnist --model_backbone mobilenet --grouping_volatility --debug
```
The hyperparameters used to train TRUNK on the specific dataset are loaded into the training script from the hyperparameters.yaml file which is found in the Datasets/dataset name/model backbone. 

## Evaluation
To evaluate TRUNK on a particular dataset, run:

```eval
$ python main.py --infer --dataset emnist --model_backbone mobilenet --grouping_volatility 
```
The hyperparameters used to conduct inference using TRUNK on the specific dataset are loaded into the test script from the hyperparameters.yaml file which is found in the Datasets/dataset name. 

## Metrics
To measure the memory size, number of floating point operations (FLOPs), number of trainable parameters, to visualize the tree for a specific dataset, to compare the ASL of an untrained root and a trained root node, and to get the sigmoid membership for a category from a trained node, execute the metrics.py script as follows

```bash
$ python metrics.py --dataset emnist --model_backbone mobilenet --visualize --untrained_asl
```

## Results
TRUNK achieves the following performance on each of the datasets as shown below. The memory and G-FLOPs are measured on the shortest and longest branch of the tree.

| Dataset Name       | Pre-Trained Weights                                                               | Inference Accuracy [%]| Latency [ms]| Memory [MB]  | G-FLOPs    |
| ------------------ |-----------------------------------------------------------------------------------| --------------------- |-----------------------------------| -----------  |------------|
| EMNIST             | [EMNIST Pre-Trained Weights](Datasets/emnist/mobilenet/1.2/model_weights/root.pt) |     84.30            | 102.50                             | 0.23 - 0.76  |0.04 - 0.37 |
| CIFAR10            | [CIFAR10 Pre-Trained Weights](Datasets/cifar10/vgg/1.02/model_weights/root.pt)    |    81.53              |    31.20  | 1.97 - 2.85  | 0.08 - 0.09 |
| SVHN               | [SVHN Pre-Trained Weights](Datasets/svhn/mobilenet/0.7/model_weights/root.pt)         |    90.24             | 77.52                         | 0.23 - 0.76  |0.04 - 0.40 |

[1]: https://arxiv.org/pdf/1801.04381.pdf
[2]: https://arxiv.org/pdf/1409.1556.pdf 