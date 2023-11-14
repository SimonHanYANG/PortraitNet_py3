# PortraitNet

Code for the paper ["PortraitNet: Real-time portrait segmentation network for mobile device".](https://www.sciencedirect.com/science/article/pii/S0097849319300305) @ CAD&Graphics 2019

---
## Introduction of PortraitNet Original Model

We propose a real-time portrait segmentation model, called PortraitNet, that can run effectively and efficiently on mobile device. PortraitNet is based on a lightweight U-shape architecture with two auxiliary losses at the training stage, while no additional cost is required at the testing stage for portrait inference. 

---
## Experimental setup

### Requirements
- python 3.9
- CUDA 12.0
- `conda create -n port python=3.9`
- `pip install -r requirements.txt`
> If you want to use another pip sources in China:
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

> Or another pip sources:
> - 中国科学技术大学 : https://pypi.mirrors.ustc.edu.cn/simple
> - 豆瓣：http://pypi.douban.com/simple/
> - 阿里云：http://mirrors.aliyun.com/pypi/simple/ 

---
### Download datasets
- [EG1800](https://pan.baidu.com/s/1myEBdEmGz6ufniU3i1e6Uw) Since several image URL links are invalid in the [original EG1800 dataset](http://xiaoyongshen.me/webpage_portrait/index.html), we finally use 1447 images for training and 289 images for validation. 

- [Supervise-Portrait](https://pan.baidu.com/s/1uBtCsLj156e_iy3DtkvjQQ) Supervise-Portrait is a portrait segmentation dataset collected from the public human segmentation dataset [Supervise.ly](https://supervise.ly/) using the same data process as EG1800.

---
## Training

```
python train.py --batchsize 32 
```

---
## Testing

Test for the image and video:
- Test for a single image
`python test.py`
- Test for a video
`Coming soon...`

---
## Visualization
Using tensorboard to visualize the training process:
```
cd path_to_save_model
tensorboard --logdir='./log'
```

---
## Download models for original PortraitNet
from Dropbox:
- [mobilenetv2_eg1800_with_two_auxiliary_losses](https://cuhko365-my.sharepoint.com/:f:/g/personal/223010087_link_cuhk_edu_cn/Emo4pKMY1MpDqN9_t44jy-sB7_V5LWIgjWqLMwVLxUddhA?e=qg6Ex7)(Training on EG1800 with two auxiliary losses)
