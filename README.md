# Efficient Featurized Image Pyramid Network for Single Shot Detector

By Yanwei Pang*, Tiancai Wang*, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao

'*' denotes equal contribution

### Introduction
Single-stage object detectors have recently gained popularity due to their combined advantage of high detection accuracy and real-time speed.
However, while promising results have been achieved by these detectors on standard-sized objects, their performance on small objects is far 
from satisfactory. To  detect very small/large objects, classical pyramid representation can be exploited, where an image pyramid is used to 
build a feature pyramid (featurized image pyramid), enabling detection across a range of scales. Existing single-stage detectors avoid such 
a featurized image pyramid representation due to its memory and time complexity. In this paper, we introduce a light-weight architecture to
efficiently produce featurized image pyramid in a single-stage detection framework.

*Note*: The speed here is tested on the Pytorch 0.3, Python 3.5 and Cuda9.0.


### Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
4. [Evaluation](#evaluation)
5. [Models](#models)

## Installation
- Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

- Install [PyTorch-0.3.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.

- Compile the nms and coco tools:
```Shell
./make.sh
```

- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below and install opencv. 
```Shell
conda install opencv
```

## Datasets
To make things easy, we provide simple VOC and COCO dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

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
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. Following the [instructions](https://github.com/rbgirshick/py-faster-rcnn/blob/77b773655505599b94fd8f3f9928dbf1a9a776c7/data/README.md) to prepare *minival2014* and *valminusminival2014* annotations. All label files (.json) should be under the COCO/annotations/ folder. It should have this basic structure
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2015/
$COCO/images/train2014/
$COCO/images/val2014/
```
*UPDATE*: The current COCO dataset has released new *train2017* and *val2017* sets which are just new splits of the same image sets. 


- Note:
  * -d: choose datasets, VOC or COCO.
  * -s: image size, 300 or 512.

## Evaluation
To evaluate a trained network:

```Shell
python test_LFIP.py -d VOC -s 300 --trained_model /path/to/model/weights
```
By default, it will directly output the mAP results on VOC2007 *test* or COCO *minival2014*. For COCO *test-dev* results, you can manually change the datasets in the `test_LFIP.py` file, then save the detection results and submitted to the server. 

## Models

* PASCAL VOC [BaiduYun Driver](https://pan.baidu.com/)
* COCO [BaiduYun Driver](https://pan.baidu.com/)

### Citing LFIP
Please cite our paper in your publications if it helps your research:

    @article{pang2019LFIP,
        title = {Efficient Featurized Image Pyramid Network for Single Shot Detection},
        author = {Yanwei Pang*, Tiancai Wang*, Rao Muhammad Anwer, Fahad Shahbaz Khan, Ling Shao},
        booktitle = {CVPR},
        year = {2019}
    }
