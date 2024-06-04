[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=tr3d-towards-real-time-indoor-3d-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-sun-rgbd-val)](https://paperswithcode.com/sota/3d-object-detection-on-sun-rgbd-val?p=tr3d-towards-real-time-indoor-3d-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-s3dis)](https://paperswithcode.com/sota/3d-object-detection-on-s3dis?p=tr3d-towards-real-time-indoor-3d-object)

## TR3D-3dof: Towards Real-Time Indoor 3D Object Detection for 3-dof rotation

This repository contains an implementation of TR3D, a 3D object detection method introduced in the paper:

> **TR3D: Towards Real-Time Indoor 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung Research<br>
> https://arxiv.org/abs/2302.02858

The following implementation of TR3D accounts for all three rotation in all 3 dimensions (axes), i.e. yaw, pitch and roll.

### Installation


You can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [getting_started.md](docs/en/getting_started.md), including MinkowskiEngine installation, replacing `open-mmlab/mmdetection3d` with `samsunglabs/tr3d`.


Most of the `TR3D`-related code locates in the following files: 
[detectors/mink_single_stage.py](mmdet3d/models/detectors/mink_single_stage.py),
[detectors/tr3d_ff.py](mmdet3d/models/detectors/tr3d_ff.py),
[dense_heads/tr3d_head.py](mmdet3d/models/dense_heads/tr3d_head.py),
[necks/tr3d_neck.py](mmdet3d/models/necks/tr3d_neck.py).



<details>
  <summary>Click to expand installation trials</summary>

| Software                 | Version           | Status                                 |
|--------------------------|-------------------|-----------------------------------------|
| CUDA                     | 11.3              | Cannot Install : Due to the driver version mismatch |
| CUDA                     | 11.7              | Cannot Install : Due to the driver version mismatch |
| CUDA                     | 12.3              | Working                                 |
| Pytorch                  | 1.12.1 + cu 11.3  | Working                                 |
| cudatoolkit              | 11.7              | Working                                 |
| cudatoolkit              | 11.3              | Cannot Install X : Due to unmet dependencies |
| Minkowski Engine         | 0.5.3             | Working                                 |
| gcc and g++ (important)  | 9.5.0             | Working  
</details>

#### Steps to install TR3D package and Minkowski engine
Create a conda env with python 3.8
```bash
conda create -n tr3d python=3.8
```

Use the above version of packages and install them. You may refer to https://github.com/SamsungLabs/tr3d/blob/main/docker/Dockerfile  for the versions of all the packages.

Set the gcc and g++ to be of version 9.5.0 (or upto 10)

Make sure nvcc is installed

```bash
nvcc --version
```

Clone the Minkowski Engine repository and make sure to follow the below instructions


```bash
export CUDA_HOME=/usr/local/cuda-11.x
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
# Or if you want local MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

The code that worked for my system (Ubuntu 20.04 with CUDA 12.1)
```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
```

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We follow the mmdetection3d data preparation protocol described in [scannet](data/scannet), [sunrgbd](data/sunrgbd), and [s3dis](data/s3dis).

For physion data preperation refer to [physion](physion).

**Setting up config files**

One can refer to `configs/tr3d/tr3d_physion_fly_config.py` as a template.


**Training**

To start training, run [train](tools/train.py) with TR3D [configs](configs/tr3d):
```shell
python tools/train.py configs/tr3d/tr3d_physion_fly_config.py
```
To train without validation:
```shell
python tools/train.py configs/tr3d/tr3d_physion_fly_config.py --no-validate
```

**Testing**

Test pre-trained model using [test](tools/dist_test.sh) with TR3D [configs](configs/tr3d):
```shell
python tools/test.py configs/tr3d/tr3d_physion_fly_config.py \
    work_dirs/<location of trained model>.pth --eval mAP
```

**Visualization**

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.3`:
```shell
python tools/test.py configs/tr3d/tr3d_scannet-3d-18class.py \
    work_dirs/tr3d_scannet-3d-18class/latest.pth --show \
    --show-dir work_dirs/<location to store output visualizations>
```

The above command stores the output in the location provided.

To view the output visualization:
```shell
python physion/output_visualizer.py pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom_unstable_0014<(location of file)>
```

### Models

The metrics are obtained in 5 training runs which utilizes the `support` type of videos. 
The runs are on a single Nvidia RTX 3080Ti (12GB) GPU. The access the models will be shortly updated

**TR3D 3D Detection**

| Dataset | mAP@0.25 | mAP@0.5 | Scenes <br> per sec.| Download |
|:-------:|:--------:|:-------:|:-------------------:|:--------:|
| ScanNet | 72.9 (72.0) | 59.3 (57.4) | 23.7 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet.log.json) &#124; [config](configs/tr3d/tr3d_scannet-3d-18class.py) |
| SUN RGB-D | 67.1 (66.3) | 50.4 (49.6) | 27.5 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_sunrgbd.log.json) &#124; [config](configs/tr3d/tr3d_sunrgbd-3d-10class.py) |
| S3DIS | 74.5 (72.1) | 51.7 (47.6) | 21.0 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_s3dis.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_s3dis.log.json) &#124; [config](configs/tr3d/tr3d_s3dis-3d-5class.py) |
| S3DIS <br> ScanNet-pretrained | 75.9 (75.1) | 56.6 (54.8) | 21.0 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet-pretrain_s3dis.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet-pretrain_s3dis.log) &#124; [config](configs/tr3d/tr3d_scannet-pretrain_s3dis-3d-5class.py) |

**RGB + PC 3D Detection on SUN RGB-D**

| Model | mAP@0.25 | mAP@0.5 | Scenes <br> per sec.| Download |
|:-----:|:--------:|:-------:|:-------------------:|:--------:|
| ImVoteNet | 63.4 | - | 14.8 | [instruction](configs/imvotenet) |
| VoteNet+FF | 64.5 (63.7) | 39.2 (38.1) | - | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/votenet_ff_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/votenet_ff_sunrgbd.log.json) &#124; [config](configs/votenet/votenet-ff_16x8_sunrgbd-3d-10class.py) |
| TR3D+FF | 69.4 (68.7) | 53.4 (52.4) | 17.5 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_ff_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_ff_sunrgbd.log.json) &#124; [config](configs/tr3d/tr3d-ff_sunrgbd-3d-10class.py) |

### Example Detections

<p align="center"><img src="./resources/github.png" alt="drawing" width="90%"/></p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@misc{rukhovich2023tr3d,
  doi = {10.48550/ARXIV.2302.02858},
  url = {https://arxiv.org/abs/2302.02858},
  author = {Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  title = {TR3D: Towards Real-Time Indoor 3D Object Detection},
  publisher = {arXiv},
  year = {2023}
}
```
