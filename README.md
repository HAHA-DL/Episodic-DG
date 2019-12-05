# Episodic-DG
This is the repo for reproducing the results in the paper Episodic Training for Domain Generalization.

##### Data
Please download the data from https://drive.google.com/open?id=0B6x7gtvErXgfUU1WcGY5SzdwZVk and use the official train/val split. \
##### ImageNet pretrained model
We use the pytorch pretrained ResNet-18 model from https://download.pytorch.org/models/resnet18-5c106cde.pth

## Enviroments

verified on
> GPU GeForce RTX 2080 Ti \
> pytorch 1.0.0 \
> Python 3.7.3 \
> Ubuntu 16.04.6

| Method  | Art | Cartoon | Photo | Sketch | Ave. |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| AGG  |76.1	|75.2	|94.9	|69.7	| 79.0|
| Epi-FCR  | 79.6|	76.8|	93.7|	77.1|	81.8|

and 

> GPU TITAN X (Pascal) \
> pytorch 0.4.1 \
> Python 2.7 \
> Scientific Linux 7.6

| Method  | Art | Cartoon | Photo | Sketch | Ave. |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| AGG  |77.6	|73.9	|94.4	|70.3	| 79.1|
| Epi-FCR  | 82.1|	77.0|	93.9|	73.0|	81.5|
