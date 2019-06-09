# Efficient Featurized Image Pyramid Network for Single Shot Detector

By Yanwei Pang†, Tiancai Wang†, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao

*†* denotes equal contribution

### Introduction
Single-stage object detectors have recently gained popularity due to their combined advantage of high detection accuracy and real-time speed.
However, while promising results have been achieved by these detectors on standard-sized objects, their performance on small objects is far 
from satisfactory. To  detect very small/large objects, classical pyramid representation can be exploited, where an image pyramid is used to 
build a feature pyramid (featurized image pyramid), enabling detection across a range of scales. Existing single-stage detectors avoid such 
a featurized image pyramid representation due to its memory and time complexity. In this paper, we introduce a light-weight architecture to
efficiently produce featurized image pyramid in a single-stage detection framework.

*Note*: The speed here is tested on the Pytorch 0.3.1, Python 3.5.6 and Cuda9.0.

## Installation
- Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

```Shell
LFIP_ROOT=/path/to/clone/LFIP
git clone https://github.com/vaesl/LFIP $LFIP_ROOT
```

- Install [PyTorch-0.3.1](http://pytorch.org/) by the following commands with Anaconda environment.

```Shell
conda create -n LFIP python=3.5
```

```Shell
source activate LFIP
```

```Shell
conda install pytorch=0.3.1 torchvision -c pytorch
```

- Install opencv. 
```Shell
conda install opencv
```

- Compile the nms and coco tools:
```Shell
./make.sh
```

## Download
To evaluate the performance reported in the paper, you need to first download the Pascal VOC and COCO dataset as well as
our trained models.

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### COCO Dataset
Download the MS COCO dataset to /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. It should have the following structure 
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2015/
$COCO/images/train2014/
$COCO/images/val2014/
```

### Trained Models

* PASCAL VOC [BaiduYun Driver](https://pan.baidu.com/)
* COCO [BaiduYun Driver](https://pan.baidu.com/)

## Evaluation
To evaluate the performance reported in the CVPR paper:

```Shell
python test_LFIP.py -d VOC -s 300 --trained_model /path/to/model/weights
```

- Note:
  * -d: choose datasets, VOC or COCO.
  * -s: image size, 300 or 512.

By default, it will directly output the mAP results on VOC2007 *test* or COCO *minival2014*. For COCO *test-dev* results, you can manually change the datasets in the `test_LFIP.py` file, then save the detection results and submitted to the server. 

### Citation
Please cite our paper in your publications if it helps your research:

    @article{Pang2019LFIP,
        title = {Efficient Featurized Image Pyramid Network for Single Shot Detection},
        author = {Yanwei Pang, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao},
        booktitle = {CVPR},
        year = {2019}
    }
