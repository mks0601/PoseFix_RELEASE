# PoseFix: Model-agnostic General Human Pose Refinement Network

<p align="center">
<b><i><font size="7">PoseFix makes pose result of any other method better from a single `.json` file!</font></i></b>
<img src="https://cv.snu.ac.kr/research/PoseFix/figs/qualitative1.PNG" width="400" height="250"> <img src="https://cv.snu.ac.kr/research/PoseFix/figs/qualitative2.PNG" width="400" height="250">
</p>

## Introduction

This repo is official **[TensorFlow](https://www.tensorflow.org)** implementation of **[PoseFix: Model-agnostic General Human Pose Refinement Network (CVPR 2019)](https://arxiv.org/abs/1812.03595)** for **model-agnostic human pose refinement** from a single RGB image. 
**What this repo provides:**
* [TensorFlow](https://www.tensorflow.org) implementation of [PoseFix: Model-agnostic General Human Pose Refinement Network](https://arxiv.org/abs/1812.03595).
* Flexible and simple code.
* Compatibility for most of the publicly available 2D multi-person pose estimation datasets including **[MPII](http://human-pose.mpi-inf.mpg.de/), [PoseTrack 2018](https://posetrack.net/), and [MS COCO 2017](http://cocodataset.org/#home)**.
* Human pose estimation visualization code (modified from [Detectron](https://github.com/facebookresearch/Detectron)).


## Dependencies
* [TensorFlow](https://www.tensorflow.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* [COCO API](https://github.com/cocodataset/cocoapi)

This code is tested under Ubuntu 16.04, CUDA 9.0, cuDNN 7.1 environment with two NVIDIA 1080Ti GPUs.

Python 3.6.5 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- data
|-- lib
|-- main
|-- tool
`-- output
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `lib` contains kernel codes for 2d multi-person pose estimation system.
* `main` contains high-level codes for training or testing the network.
* `tool` contains dataset converter. I set MS COCO as reference format and provide `mpii2coco` and `posetrack2coco` converting code.
* `output` contains log, trained models, visualized outputs, and test result.

### Data
You need to follow directory structure of the `data` as below.
```
${POSE_ROOT}
|-- data
|-- |-- MPII
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg
|-- |-- PoseTrack
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train2018.json
|       |   |-- val2018.json
|       |   `-- test2018.json
|       |-- original_annotations
|       |   |-- train/
|       |   |-- val/
|       |   `-- test/
|       `-- images
|           |-- train/
|           |-- val/
|           `-- test/
|-- |-- COCO
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- person_keypoints_train2017.json
|       |   |-- person_keypoints_val2017.json
|       |   `-- image_info_test-dev2017.json
|       `-- images
|           |-- train2017/
|           |-- val2017/
|           `-- test2017/
`-- |-- imagenet_weights
|       |-- resnet_v1_50.ckpt
|       |-- resnet_v1_101.ckpt
|       `-- resnet_v1_152.ckpt
```
* In the `tool`, run `python mpii2coco.py` to convert MPII annotation files to MS COCO format (`MPII/annotations`).
* In the `tool`, run `python posetrack2coco.py` to convert PoseTrack annotation files to MS COCO format (`PoseTrack/annotations`).
* Download imagenet pre-trained resnet models from [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim) and place it in the `data/imagenet_weights`.
* Except for `annotations` of the MPII and PoseTrack, all other directories are original version of downloaded ones.
* If you want to add your own dataset, you have to convert it to [MS COCO format](http://cocodataset.org/#format-data).
* You can change default directory structure of `data` by modifying `dataset.py` of each dataset folder.

### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- log
|-- |-- model_dump
|-- |-- result
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.
* You can change default directory structure of `output` by modifying `main/config.py`.

## Running PoseFix
### Start
* Run `pip install -r requirement.txt` to install required modules.
* Run `cd ${POSE_ROOT}/lib` and `make` to build NMS modules.
* In the `main/config.py`, you can change settings of the model including dataset to use, network backbone, and input size and so on.

### Train
`input_pose/test_on_trainset/result.json` should be prepared before training. This is test result on the training set and used when synthesizing input pose of not annotated keypoints in the training stage. I used testing result of [TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose). 

In the `main` folder, run
```bash
python train.py --gpu 0-1
```
to train the network on the GPU 0,1. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0-1 --continue
```
`--gpu 0,1` can be used instead of `--gpu 0-1`.

### Test
`input_pose/name_of_input_pose.json` is pose estimation result of any other method. You have to rename the filename and also `input_pose_path` of the `data/$DATASET/dataset.py`. The input pose file should be follow [MS COCO format](http://cocodataset.org/#format-results).

Place trained model at the `output/model_dump/$DATASET/` and pose estimation result of any other method (`name_of_input_pose.json`) to `data/$DATASET/input_pose/`.

In the `main` folder, run 
```bash
python test.py --gpu 0-1 --test_epoch 140
```
to test the network on the GPU 0,1 with 140th epoch trained model. `--gpu 0,1` can be used instead of `--gpu 0-1`.

## Results
Here I report the performance of the PoseFix. Also, I provide pre-trained models of the PoseFix and `test_on_trainset/result.json`.
 
As this repo outputs compatible output files for MS COCO and PoseTrack, you can directly use [cocoapi](https://github.com/cocodataset/cocoapi) or [poseval]( https://github.com/leonid-pishchulin/poseval) to evaluate result on the MS COCO or PoseTrack dataset. You have to convert the produced `mat` file to MPII `mat` format to evaluate on MPII dataset following [this](http://human-pose.mpi-inf.mpg.de/#evaluation).

### Results on MSCOCO 2017 dataset
<p align="center">
<img src="https://cv.snu.ac.kr/research/PoseFix/figs/ap_improvement_coco.PNG">
</p>

* You have to set `dataset`, `backbone` and `input_shape` to those of the model in `config.py`.
* Pre-trained PoseFix model (COCO 2017, ResNet-152, 384x288) [[model](https://cv.snu.ac.kr/research/PoseFix/COCO2017/model/PoseFix_coco2017_resnet152_384x288.zip)]
* Testing result on the COCO 2017 training set of [TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose) [[kps](https://cv.snu.ac.kr/research/PoseFix/COCO2017/test_on_trainset/result.json)]

### Results on PoseTrack 2018 dataset
<p align="center">
<img src="https://cv.snu.ac.kr/research/PoseFix/figs/ap_improvement_posetrack.PNG">
</p>

* You have to set `dataset`, `backbone` and `input_shape` to those of the model in `config.py`.
* Pre-trained PoseFix model (PoseTrack 2018, ResNet-152, 384x288) [[model](https://cv.snu.ac.kr/research/PoseFix/PoseTrack2018/model/PoseFix_posetrack2018_resnet152_384x288.zip)]
* Testing result on the PoseTrack 2018 training set of [TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose) [[kps](https://cv.snu.ac.kr/research/PoseFix/PoseTrack2018/test_on_trainset/result.json)]

## Acknowledgements
This repo is largely modified from [TensorFlow repo of CPN](https://github.com/chenyilun95/tf-cpn) and [PyTorch repo of Simple](https://github.com/Microsoft/human-pose-estimation.pytorch).

## Reference
  ```
@InProceedings{Moon_2019_CVPR_PoseFix,
  author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
  title = {PoseFix: Model-agnostic General Human Pose Refinement Network},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

