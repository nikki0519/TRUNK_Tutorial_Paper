# ModelComparisons
PyTorch code provided to conduct experiments and compare the TRUNK architecture on metrics such as validation accuracy, inference runtime per image, memory requirements, number of floating point operations (FLOPs), and the number of trainable parameters, to well-known monolithic architectures:
1. [VGG-16][1]
2. [ResNet-50][2]
3. ResNet-50 Quantize Aware Training (QAT)
4. [MobileNetv2][3]
5. [ConvNeXt-Base][4]
6. [Vision Transformers (ViT)][5]
7. [DinoV2][6]
8. [DeIT][7]

To reproduce the comparison results on the EMNIST, CiFAR10, and SVHN datasets, execute the comparisons.py script. The data will be downloaded when this script is executed. Pre-Trained weights are also available in the respective folders.

## Training and Testing

```bash
python comparisons.py --dataset emnist --model resnet
```

## Results
The results for each dataset are as follows:

### EMNIST
| Model name         | Pre-Trained Weights  | Inference Accuracy [%] | Inference Time per Image [s/Image] | Memory [MB] | G-Flops |
| ------------------ |--------------------- | ---------------------- | -----------------------------------| ------------| --------|
| ResNet50   |  [ResNet50 Pre-Trained Weights](ResNet/resnet_weights_emnist.pt)           |  87.60          | 6.23 | 94.71 | 4.05 |
| ResNet50 Quantized   |  [ResNet50 Quantized Pre-Trained Weights](ResNetQuantized/resnet_quantized_weights_emnist.pt) |  88.93  | 75.17 | 95.48 | 0.02 |
| VGG16 Pruned   |  [VGG16 Pruned Pre-Trained Weights](VGGPruned/emnist/prune/emnist-local-l1-vgg16/emnist_vgg16_l1.pth) | 88.21  | 1.19 | 29.52 | 9.74 |
| VGG16  |  [VGG-16 Pre-Trained Weights](VGG/vgg_weights_emnist.pt) |  87.27  | 2.43 | 537.82 | 15.41 |
| ConvNeXt-Base   |  [ConvNeXt-Base Pre-Trained Weights](ConvNeXt/convnext_weights_emnist.pt) |  89.45  | 8.29 | 350.57 | 15.36  |
| MobileNetv2   |  [MobileNetv2 Pre-Trained Weights](MobileNet/mobilenet_weights_emnist.pt) | 87.34  | 4.54 | 9.38 | 0.32 |
| ViT   |  [ViT Pre-Trained Weights](ViT/vit_weights_emnist.pt) |  81.33  | 6.18 | 341.83 | 11.21 |
| DinoV2   |  [DinoV2 Pre-Trained Weights](DinoV2/dinov2_weights_emnist.pt) |  81.31  | 11.95 | 345.19 | 22.23 |
| DeIT   |  [DeIT Pre-Trained Weights](DeIT/deit_weights_emnist.pt) |  88.30  | 4.54 | 21.80 | 1.08 |

### CIFAR10
| Model name         | Pre-Trained Weights  | Inference Accuracy [%] | Inference Time per Image [s/Image] | Memory [MB] | G-Flops |
| ------------------ |--------------------- | ---------------------- | -----------------------------------| ------------| --------|
| ResNet50   |  [ResNet50 Pre-Trained Weights](ResNet/resnet_weights_cifar10.pt)           |  85.09          | 6.23 | 94.43 | 4.13 |
| ResNet50 Quantized   |  [ResNet50 Quantized Pre-Trained Weights](ResNetQuantized/resnet_quantized_weights_cifar10.pt) |  85.46  | 76.51 | 95.20 | 0.02 |
| VGG16 Pruned   |  [VGG16 Pruned Pre-Trained Weights](VGGPruned/cifar10/prune/cifar10-local-l1-vgg16/cifar10_vgg16_l1.pth) |  92.40  | 0.04 | 29.52 | 0.20 |
| VGG16  |  [VGG-16 Pre-Trained Weights](VGG/vgg_weights_cifar10.pt) |  86.88  | 3.09 | 537.22 | 15.47 |
| ConvNeXt-Base   |  [ConvNeXt-Base Pre-Trained Weights](ConvNeXt/convnext_weights_cifar10.pt) |  93.29  | 9.33 | 350.43 | 15.37  |
| MobileNetv2   |  [MobileNetv2 Pre-Trained Weights](MobileNet/mobilenet_weights_cifar10.pt) | 84.54  | 4.79 | 9.19 | 0.33 |
| ViT   |  [ViT Pre-Trained Weights](ViT/vit_weights_cifar10.pt) |  90.55  | 6.47 | 343.28 | 11.29 |
| DinoV2   |  [DinoV2 Pre-Trained Weights](DinoV2/dinov2_weights_cifar10.pt) |  85.64  | 12.16 | 346.40 | 22.30 |
| DinoV2   |  [DeIT Pre-Trained Weights](DeIT/deit_weights_cifar10.pt) |  89.48  | 4.38 | 22.17 | 1.06 |

### SVHN
| Model name         | Pre-Trained Weights  | Inference Accuracy [%] | Inference Time per Image [s/Image] | Memory [MB] | G-Flops |
| ------------------ |--------------------- | ---------------------- | -----------------------------------| ------------| --------|
| ResNet50   |  [ResNet50 Pre-Trained Weights](ResNet/resnet_weights_svhn.pt)           |  94.43          | 6.15 | 94.43 | 4.13 |
| ResNet50 Quantized   |  [ResNet50 Quantized Pre-Trained Weights](ResNetQuantized/resnet_quantized_weights_svhn.pt) |  94.01  | 73.91 | 95.20 | 0.02 |
| VGG16 Pruned   |  [VGG16 Pruned Pre-Trained Weights](VGGPruned/svhn/prune/svhn-local-l1-vgg16/svhn_vgg16_l1.pth) |  94.12 | 1.38 | 29.52 | 9.78 |
| VGG16  |  [VGG-16 Pre-Trained Weights](VGG/vgg_weights_svhn.pt) |  94.63  | 2.53 | 537.21 | 15.47 |
| ConvNeXt-Base   |  [ConvNeXt-Base Pre-Trained Weights](ConvNeXt/convnext_weights_svhn.pt) |  97.03  | 8.28 | 350.43 | 15.37  |
| MobileNetv2   |  [MobileNetv2 Pre-Trained Weights](MobileNet/mobilenet_weights_svhn.pt) | 95.69  | 4.62 | 9.19 | 0.33 |
| ViT   |  [ViT Pre-Trained Weights](ViT/vit_weights_svhn.pt) |  95.87  | 6.32 | 343.28 | 11.29 |
| DinoV2   |  [DinoV2 Pre-Trained Weights](DinoV2/dinov2_weights_svhn.pt) |  94.32  | 11.96 | 345.19 | 22.23 |
| DinoV2   |  [DeIT Pre-Trained Weights](DeIT/deit_weights_svhn.pt) |  95.19  | 4.44 | 22.16 | 1.08 |

[1]: https://arxiv.org/pdf/1409.1556.pdf
[2]: https://arxiv.org/pdf/1512.03385.pdf
[3]: https://arxiv.org/pdf/1801.04381.pdf
[4]: https://arxiv.org/pdf/2201.03545.pdf
[5]: https://arxiv.org/pdf/2010.11929.pdf
[6]: https://arxiv.org/pdf/2304.07193.pdf
[7]: https://arxiv.org/pdf/2012.12877.pdf