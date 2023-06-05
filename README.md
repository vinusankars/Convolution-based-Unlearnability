# CUDA: Convolution-based Unlearnable Datasets

Requirements
------------

Python 3.8.5 (GCC 7.3.0)
NVIDIA GPU with CUDA 11.0
Python requirements in requirements.txt


Directory tree
--------------

The readme file is in the current directory "."
Make folder "../datasets/" where datasets will be downloaded
Make folder "results/" where results will be saved


Codes
-----
{densenet, resnet, vgg}.py contain networks from https://github.com/fshp971/robust-unlearnable-examples/tree/main/models
util.py contains progress bar utils from https://github.com/HanxunH/Unlearnable-Examples
final_filter_unlearnable.py contains code for executing CUDA dataset training.
final_muladv.py contains code for executing Deconvolution-based Adversarial Training (DAT) on CUDA CIFAR-10 dataset with ResNet-18.


To Run
------

For executing final_filter_unlearnable.py goto "." and run

'''
python final_filter_unlearnable.py --arch='resnet18' --dataset='cifar10' --train-type='adv' \
--blur-parameter=0.3 --seed=0 --pgd-norm='linf' --pgd-steps=10 --pgd-radius=0.015 --mix=1.0 \
--name='results/resnet18_cifar10_adv_bp=0.3_linf_eps=4_steps=10_seed0_mix=1.0.pkl'
'''

Above code will perform L_{\infty} adversarial training with CUDA CIFAR-10 dataset using ResNet-18.

For executing DAT, goto "." and run

'''
python final_muladv.py
'''
