# High-resolution networks (HRNets) for object detection
## Introduction
This is the official code of [High-Resolution Representations for Object Detection](https://arxiv.org/abs/1904.04514). We extend the high-resolution representation (HRNet) [1] by augmenting the high-resolution representation by aggregating the (upsampled) representations from all the parallel
convolutions, leading to stronger representations. We build a multi-level representation from the high resolution and apply it to the Faster R-CNN, Mask R-CNN and Cascade R-CNN framework. This proposed approach achieves superior results to existing single-model networks 
on COCO object detection. The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection)

<div align=center>

![](images/hrnetv2p.png)

</div>

## Performance
### ImageNet pretrained models
HRNetV2 ImageNet pretrained models are now available! Codes and pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Imagenet-Classification)

All models are trained on COCO *train2017* set and evaluated on COCO *val2017* set. Detailed settings or configurations are in [`configs/hrnet`](configs/hrnet).

**Note:** Models are trained with the newly released code and the results have minor differences with that in the paper. 
Current results will be updated soon and more models and results are comming.

### Faster R-CNN
|Backbone|#Params|GFLOPs|lr sched|mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 |26.2M|159.1| 1x | 36.1 | [HRNetV2-W18](https://drive.google.com/open?id=1qxyRvGHEtuDR74IGrsKLN0Si4N-S2vgq) | [FasterR-CNN-HR18-1x.pth](https://drive.google.com/open?id=1EXA85LQkHZvk0LN7F3DH4rmnWJpWXfy_)|
| HRNetV2-W18 |26.2M|159.1| 2x | 38.1 | [HRNetV2-W18](https://drive.google.com/open?id=1qxyRvGHEtuDR74IGrsKLN0Si4N-S2vgq) | [FasterR-CNN-HR18-2x.pth](https://drive.google.com/open?id=1sZTeCtO-TFtzj0l-q1Lae6k_k_CL48A2)|
| HRNetV2-W32 |45.0M|245.3| 1x | 39.5 | [HRNetV2-W32](https://drive.google.com/open?id=1EF2AUHIqbBEekL6TaMYO2M5zStdAAKJ5) | [FasterR-CNN-HR32-1x.pth](https://drive.google.com/open?id=1sZTeCtO-TFtzj0l-q1Lae6k_k_CL48A2)|
| HRNetV2-W32 |45.0M|245.3| 2x | 40.8 | [HRNetV2-W32](https://drive.google.com/open?id=1EF2AUHIqbBEekL6TaMYO2M5zStdAAKJ5) | [FasterR-CNN-HR32-2x.pth](https://drive.google.com/open?id=17LUcyoNff5j1QHRPe4vyRyNzrLVn-8AI)|
| HRNetV2-W40 |60.5M|314.9| 1x | 40.4 | [HRNetV2-W40](https://drive.google.com/open?id=1iAAZhmxkkYB_pqZ2MlAm4iCC_OBLxzC1) | [FasterR-CNN-HR40-1x.pth](https://drive.google.com/open?id=1UcPuYnBt68itZFvdxDdZfyt472k6nlzv)|
| HRNetV2-W40 |60.5M|314.9| 2x | 41.4 | [HRNetV2-W40](https://drive.google.com/open?id=1iAAZhmxkkYB_pqZ2MlAm4iCC_OBLxzC1) | [FasterR-CNN-HR40-2x.pth](https://drive.google.com/open?id=1XuljlzTGHCSQXykaNXTLaMfvhC8bJ0Hr)|

### Cascade R-CNN
**Note:** we follow the original paper[2] and adopt 280k training iterations which is equal to 20 epochs in mmdetection.

|Backbone|lr sched|mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|
| ResNet-101  | 20e | 42.8 | - | [CascadeR-CNN-R101-20e.pth](https://drive.google.com/open?id=1HEr2TMZcO7m66Xv3VILPhca2bhuhdy0l)|
| HRNetV2-W32 | 20e | 43.7 | [HRNetV2-W32](https://drive.google.com/open?id=1EF2AUHIqbBEekL6TaMYO2M5zStdAAKJ5) | [CascadeR-CNN-HR32-20e.pth](https://drive.google.com/open?id=18CrpwpfKB6o0U9l3sqJ12qu4xnJLxlB6)|

## Quick start
#### Environment
This code is developed using on Python 3.6 and PyTorch 1.0.0 on Ubuntu 16.04 with NVIDIA GPUs. Training and testing are 
performed using 4 NVIDIA P100 GPUs with CUDA 9.0 and cuDNN 7.0. Other platforms or GPUs are not fully tested.

#### Install
1. Install PyTorch 1.0 following the [official instructions](https://pytorch.org/)
2. Install `mmcv`
````bash
pip install mmcv
````
3. Install `pycocotools`
````bash
git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install \
 && cd ../../
````
4. Install `mmdetection-hrnet`
````bash
git clone https://github.com/HRNet/HRNet-Object-Detection.git

cd mmdetection-hrnet
# compile CUDA extensions.
chmod +x compile.sh
./compile.sh

# run setup
python setup.py install 

# or install locally
python setup.py install --user
````
For more details, see [INSTALL.md](INSTALL.md)

#### HRNetV2 pretrained models
```bash
cd mmdetection-hrnet
# Download pretrained models into this folder
mkdir hrnetv2_pretrained
```
#### Datasets
Please download the COCO dataset from [cocodataset](http://cocodataset.org/#download). If you use `zip` format, please specify `CocoZipDataset` in **config** files or `CocoDataset` if you unzip the downloaded dataset. 

#### Train (multi-gpu training)
Please specify the configuration file in `configs` (learning rate should be adjusted when the number of GPUs is changed).
````bash
python -m torch.distributed.launch --nproc_per_node <GPUS NUM> tools/train.py <CONFIG-FILE> --launcher pytorch
# example:
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py --launcher pytorch
````

#### Test
````bash
python tools/test.py <CONFIG-FILE> <MODEL WEIGHT> --gpus <GPUS NUM> --eval bbox --out result.pkl
# example:
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py work_dirs/faster_rcnn_hrnetv2p_w18_1x/model_final.pth --gpus 4 --eval bbox --out result.pkl
````

**NOTE:** If you meet some problems, you may find a solution in [issues of official mmdetection repo](https://github.com/open-mmlab/mmdetection/issues) 
 or submit a new issue in our repo.
 
## Other applications of HRNets (codes and models):
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) (coming soon)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) (coming soon)
* [Image classification](https://github.com/HRNet/HRNet-Imagenet-Classification)
 
## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019.

[2] Cascade R-CNN: Delving into High Quality Object Detection. Zhaowei Cai, and Nuno Vasconcetos. CVPR 2018.

## Acknowledgement
Thanks [@open-mmlab](https://github.com/open-mmlab) for providing the easily-used code and kind help!
