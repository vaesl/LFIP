# Efficient Featurized Image Pyramid Network for Single Shot Detector

By Yanwei Pang†, Tiancai Wang†, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao

† denotes equal contribution

### Introduction
Single-stage object detectors have recently gained popularity due to their combined advantage of high detection accuracy and real-time speed.
However, while promising results have been achieved by these detectors on standard-sized objects, their performance on small objects is far 
from satisfactory. To  detect very small/large objects, classical pyramid representation can be exploited, where an image pyramid is used to 
build a feature pyramid (featurized image pyramid), enabling detection across a range of scales. Existing single-stage detectors avoid such 
a featurized image pyramid representation due to its memory and time complexity. In this paper, we introduce a light-weight architecture to
efficiently produce featurized image pyramid in a single-stage detection framework.

## Installation
- Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [RFBNet](https://github.com/ruinmessi/RFBNet).

```Shell
    LFIP_ROOT=/path/to/clone/LFIP
    git clone https://github.com/vaesl/LFIP $LFIP_ROOT
```
- The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.5/6 and [PyTorch]((http://pytorch.org/)) v0.3.1. 
NVIDIA GPUs are needed for testing. After install Anaconda, create a new conda environment, activate the environment and install pytorch0.3.1.

```Shell
    conda create -n LFIP python=3.5
    source activate LFIP
    conda install pytorch=0.3.1 torchvision -c pytorch
```

- Install opencv. 
```Shell
    conda install opencv
```

- Compile both [COCOAPI](https://github.com/cocodataset/cocoapi) and NMS:
```Shell
    cd $LFIP_ROOT/
    ./make.sh
```

## Download
To evaluate the performance reported in the paper, Pascal VOC and COCO dataset as well as our trained models need to be downloaded.

### VOC Dataset
- Directly download the images and annotations from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/) and put them into $LFIP_ROOT/data/VOCdevkit/.
- Create the `VOCdevkit` folder and make the data(or create symlinks) folder like:

  ~~~
  ${$LFIP_ROOT}
  |-- data
  `-- |-- VOCdevkit
      `-- |-- VOC2007
          |   |-- annotations
          |   |-- ImageSets
          |   |-- JPEGImages
          |-- VOC2012
          |   |-- annotations
          |   |-- ImageSets
          |   |-- JPEGImages
          |-- results
  ~~~

### COCO Dataset
- Download the images and annotation files from coco website [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${$LFIP_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2014.json
          |   |-- instances_val2014.json
          |   |-- image_info_test-dev2015.json
          `-- images
          |   |-- train2014
          |   |-- val2014
          |   |-- test2015
          `-- cache
  ~~~

### Trained Models
Please access to [Google Driver](https://drive.google.com/drive/folders/1MSrUiGRYl9QVJgsjrTDtAAlSm7venRhl?usp=sharing) 
or [BaiduYun Driver](https://pan.baidu.com/s/1F0pqYmA8wJUED_jV8xmFNw) to obtain our trained models for 
PASCAL VOC and COCO and put the models into corresponding directory(e.g. '~/weights/COCO/LFIP_COCO_300/'). 
Note that the access code for the BaiduYun Driver is jay3 and for the time being we only release models with 300*300 input size. 

## Evaluation
To check the performance reported in the paper:

```Shell
python test_LFIP.py -d VOC -s 300 --trained_model /path/to/model/weights
```

where '-d' denotes datasets, VOC or COCO and '-s' represents image size, 300 or 512.

## Citation
Please cite our paper in your publications if it helps your research:

    @article{Pang2019LFIP,
        title = {Efficient Featurized Image Pyramid Network for Single Shot Detection},
        author = {Yanwei Pang, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao},
        booktitle = {CVPR},
        year = {2019}
    }
