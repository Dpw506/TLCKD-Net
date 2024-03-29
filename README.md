# TLCKD-Net
## Transformer With Large Convolution Kernel Decoder Network for Salient Object Detection in Optical Remote Sensing Images

The official pytorch implementation of the paper "Transformer With Large Convolution Kernel Decoder Network for Salient Object Detection in Optical Remote Sensing Images", [CUIV](https://doi.org/10.1016/j.cviu.2023.103917),2024

### Network Architecture
<img src="fig/pipeline.png" width="80%">

### Requirements
```
python=3.11.4
pytorch=2.0.1+cu117
timm=0.9.7
```

### Training and Testing
The significantly detected results of our method on the two datasets are at EORSSD([Baidu](https://pan.baidu.com/s/1OI27aSpQ76aIqP_L94qNhw))(code:2024) and ORSSD([Baidu](https://pan.baidu.com/s/1Fza3qxqMErbZP6M2K2GoEA))(code:2024), respectively.
Also, if you want to test our method, download the training weights here ([Baidu](https://pan.baidu.com/s/10g5UFdmLoY3dkaMbG3Rf_Q))(code:2024) and create a new checkpoint's file (```checkpoint/```) folder to store the weights. During this process, you also need to download the pre-trained weights （[rest_large.pth](https://pan.baidu.com/s/1ntXd8WDlnVcFiESpOTc6SA))(code:2024) for Rest on ImageNet, and place it in the folder ```./ImgNet```. Then run test.py as follows:
```
python test.py --checkpoint_dir ./checkpoint/TLKCDNet(EORSSD_ckp).pkl --imagenet_model ./ImgNet
```

### Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@article{DONG2024103917,
title = {Transformer with large convolution kernel decoder network for salient object detection in optical remote sensing images},
journal = {Computer Vision and Image Understanding},
pages = {103917},
year = {2024},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2023.103917},
url = {https://www.sciencedirect.com/science/article/pii/S1077314223002977},
author = {Pengwei Dong and Bo Wang and Runmin Cong and Hai-Han Sun and Chongyi Li},
keywords = {Salient object detection, Optical remote sensing image, Transformer, Large convolutional kernel},
abstract = {Despite salient object detection in optical remote sensing images (ORSI-SOD) has made great strides in recent years, it is still a very challenging topic due to various scales and shapes of objects, cluttered backgrounds, and diverse imaging orientations. Most previous deep learning-based methods fails to effectively capture local and global features, resulting in ambiguous localization and semantic information and inaccurate detail and boundary prediction for ORSI-SOD. In this paper, we propose a novel Transformer with large convolutional kernel decoding network, named TLCKD-Net, which effectively models the long-range dependence that is indispensable for feature extraction of ORSI-SOD. First, we utilize Transformer backbone network to perceive global and local details of salient objects. Second, a large convolutional kernel decoding module based on self-attention mechanism is designed for different sizes of salient objects to extract feature information at different scales. Then, a large convolutional refinement and a Salient Feature Enhancement Module are used to recover and refine the saliency features to obtain high quality saliency maps. Extensive experiments on two public ORSI-SOD datasets show that our proposed method outperforms 16 state-of-the-art methods both qualitatively and quantitatively. In addition, a series of ablation studies demonstrate the effectiveness of different modules for ORSI-SOD. Our source code is publicly available at https://github.com/Dpw506/TLCKD-Net.}
}
```
If you encounter any problems with the code, want to report bugs, etc.
Please contact me at tjuwb@nxu.edu.cn or d2568244421@163.com.
