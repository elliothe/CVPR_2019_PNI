# Paramertic Noise Injection for Adversarial Defense

This repository contains a Pytorch implementation of our paper titled "[Parametric Noise Injection: Trainable Randomness to Improve Deep Neural Network Robustness against Adversarial Attack]()"

If you find this project useful to you, please cite [our work]():

```bibtex
@inproceedings{he2019PNI,
 title={Parametric Noise Injection: Trainable Randomness to Improve Deep Neural Network Robustness against Adversarial Attack},
 author={He, Zhezhi and Adnan Siraj Rakin and Fan, Deliang},
 booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
 pages={},
 year={2019}
}
```

## Table of Contents
  
- [Dependencies](#Dependencies )
- [Usage](#Usage )
- [Results](#Results )
  - ResNet-20/32/44/56 on CIFAR-10
  - AlexNet and ResNet-18/34/50/101 on ImageNet
- [Methods](#Methods )
  
  
## Dependencies:
  
  
* Python 3.6 (Anaconda)
* Pytorch 4.1
  
## Set up A Conda python Environment
Anaconda allows you to have different environments installed on your computer to access different versions of `python` and different libraries. Sometimes, the conflict of library versions may causes errors and packages not working.

<!-- Use class="notice" for blue notes, class="warning" for red warnings, and class="success" for green notes. -->

<div class="Notice">
You must replace `meowmeowmeow` with your personal API key.
</div>
  
  
## Usage



## Notes for Experiments setup

1. Previous works remove the normalization layer from the data-preprocessing. Since we still expect the fast convergence benefit from the normalization of input image, we add a normalization in front of the neural network which perform the identical functionality but with the normalization incoporated within the backward computation graph.

2. 


