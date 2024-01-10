# TRUNK
PyTorch implementation and pre-trained models of TRUNK for the EMNIST, CiFAR10, and SVHN datasets. For details, see the papers: [A Tutorial Paper for Tree-based Unidirectional Neural Networks for Low Powered Computer Vision (LPCV)](#Include Link).

Despite effectively reducing the memory requirements, most low-powered computer vision techniques still retain their monolithic structures that rely on a single DNN to simultaneously classify all the categories, causing it to still conduct redundant operations. To address this issue, we introduce a tree-based unidirectional neural network (TRUNK) to eliminate redundancies for image classification. The TRUNK architecture reduces these redundant operations by using only a small subsets of the neurons in the DNN by using a tree-based approach. 

The architecture design varies by the different datasets used and is inspired by the [MobileNetv2][1] and [VGG-16][2] networks. The pre-trained weights are available for the MobileNet inspired architecture. 

The Datasets directory is further divided into the three folderes representing the three datasets (i.e. EMNSIT, CiFAR10, SVHN) we've used to train/test TRUNK. Within these folders, we have the models/ directory where the MobileNetv2 and VGG16 networks are saved. Within the mobilenet/ and vgg/ directories, the inputs and results (i.e. hyper-parameters, model weights, model softmax, details of the tree, and inference results) are stored. 

The data (EMNIST, CiFAR10, and SVHN) used to train and test the TRUNK model will be downloaded when executing the main.py script as shown below

```bash
# To train TRUNK
$ python main.py --train --dataset emnist --model_backbone mobilenet 

# To conduct inference on TRUNK
$ python main.py --infer --dataset emnist --model_backbone mobilenet
```

To measure the memory size, number of floating point operations (FLOPs), number of trainable parameters, to visualize the tree for a specific dataset, to compare the ASL of an untrained root and a trained root node, and to get the sigmoid membership for a category from a trained node, execute the metrics.py script as follows

```bash
$ python metrics.py --dataset emnist --model_backbone mobilenet --visualize --untrained_asl
```

[1]: https://arxiv.org/pdf/1801.04381.pdf
[2]: https://arxiv.org/pdf/1409.1556.pdf 