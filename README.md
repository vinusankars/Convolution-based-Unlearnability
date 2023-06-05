# CUDA: Convolution-based Unlearnable Datasets (CVPR 2023)
#### Authors: Vinu Sankar Sadasivan, Mahdi Soltanolkotabi, Soheil Feizi

Paper: https://arxiv.org/abs/2303.04278

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

```
python final_filter_unlearnable.py --arch='resnet18' --dataset='cifar10' --train-type='adv' \
--blur-parameter=0.3 --seed=0 --pgd-norm='linf' --pgd-steps=10 --pgd-radius=0.015 --mix=1.0 \
--name='results/resnet18_cifar10_adv_bp=0.3_linf_eps=4_steps=10_seed0_mix=1.0.pkl'
```

Above code will perform L_{\infty} adversarial training with CUDA CIFAR-10 dataset using ResNet-18.

For executing DAT, goto "." and run

```
python final_muladv.py
```




> COPYRIGHT AND PERMISSION NOTICE
> UMD Software [Can AI-Generated Text be Reliably Detected?] Copyright (C) 2022 University of Maryland
> All rights reserved.
> The University of Maryland (“UMD”) and the developers of [CUDA: Convolution-based Unlearnable Datasets] software (“Software”) give recipient (“Recipient”) permission to download a single copy of the Software in source code form and use by university, non-profit, or research institution users only, provided that the following conditions are met:
> 
> Recipient may use the Software for any purpose, EXCEPT for commercial benefit.
> Recipient will not copy the Software.
> Recipient will not sell the Software.
> Recipient will not give the Software to any third party.
> Any party desiring a license to use the Software for commercial purposes shall contact:
> UM Ventures, College Park at UMD at otc@umd.edu.
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS, CONTRIBUTORS, AND THE UNIVERSITY OF MARYLAND "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER, CONTRIBUTORS OR THE UNIVERSITY OF MARYLAND BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
