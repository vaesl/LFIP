import os.path

LFIP_ROOT = os.path.dirname(os.getcwd())
# access to the VOC and COCO dataset
VOCdir = os.path.join(LFIP_ROOT, "data/VOCdevkit/")
COCOdir = os.path.join(LFIP_ROOT, "data/coco/")

# LFIP CONFIGS
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}


COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}
