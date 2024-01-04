# ModelComparisons
PyTorch code provided to conduct experiments and compare our TRUNK architecture on metrics such as validation accuracy, inference runtime per image, memory requirements, number of floating point operations (FLOPs), and the number of trainable parameters, to well-known monolithic architectures:
1. [VGG-16][1]
2. [ResNet-50][2]
3. ResNet-50 Quantize Aware Training (QAT)
4. ResNet-50 Pruned
5. [MobileNetv2][3]
6. [ConvNeXt-Base][4]
7. [Vision Transformers (ViT)][5]
8. [DinoV2][6]

To reproduce the comparison results on the EMNIST, CiFAR10, and SVHN datasets, execute the comparisons.py script. The data will be downloaded when this script is executed. Pre-Trained weights are also available in the respective folders.

```bash
python comparisons.py --dataset emnist --model resnet
```

[1]: https://arxiv.org/pdf/1409.1556.pdf
[2]: https://arxiv.org/pdf/1512.03385.pdf
[3]: https://arxiv.org/pdf/1801.04381.pdf
[4]: https://arxiv.org/pdf/2201.03545.pdf
[5]: https://arxiv.org/pdf/2010.11929.pdf
[6]: https://arxiv.org/pdf/2304.07193.pdf
 