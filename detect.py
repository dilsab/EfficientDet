import matplotlib
matplotlib.use('Agg')

import torch

from backbone import EfficientDetBackbone

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_input_size(compound_coef):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    return input_sizes[compound_coef]


def detect_objects(img_path):
    weight_file =  os.path.join('logs', 'coffee_headphones_raven_imbalanced', 'efficientdet-d0_29_6270.pth')
    obj_list = ['coffee', 'headphones', 'raven']
    compound_coef = 0

    use_gpu = False
    threshold = 0.4
    iou_threshold = 0.2

    input_size = get_input_size(compound_coef)

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size, rgb=True)

    if use_gpu:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
                                 )

    if use_gpu:
        model.load_state_dict(torch.load(weight_file))
    else:
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

    model.requires_grad_(False)
    model.eval()

    if use_gpu:
        model = model.cuda()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    detection_result = { 'classes': {}, 'detected_total': 0 }

    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]['class_ids'][j]]

            if obj not in detection_result['classes']:
                detection_result['classes'][obj] = 0
            detection_result['classes'][obj] += 1
            detection_result['detected_total'] += 1
            score = float(out[i]['scores'][j])

            cv2.rectangle(ori_imgs[i], (x1, y2 - 50), (x1 + 120 + len(obj) * 22, y2), (0, 0, 0), cv2.FILLED)
            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y2 - 15), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 2)

            plt.imshow(ori_imgs[i])
    plt.axis('off')
    detection_result['image'] = plt

    return detection_result
