import math
import os
import cv2
from skimage import io
import json
from skimage.util import view_as_blocks
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


def pad_img_row(lesion_fs, data_res):
    n_lesion = len(lesion_fs)
    print(f'{len(lesion_fs)} lesions detected.')
    if n_lesion == 0:
        img_row = np.ones((data_res, data_res)) * 255
    elif n_lesion == 1:
        img_row = np.concatenate([io.imread(lesion_fs[0]), np.ones((data_res, (data_res+32)*(4-n_lesion))) * 255], axis=1)
    else:
        img_row = io.imread(lesion_fs[0])
        for i in range(1, n_lesion):
            cur_img = io.imread(lesion_fs[i])
            cur_img_with_pad = np.concatenate([np.ones((data_res, 32)) * 255, cur_img], axis=1)
            img_row = np.concatenate([img_row, cur_img_with_pad], axis=1)

        if n_lesion < 4:
            img_row = np.concatenate([img_row, np.ones((data_res, (data_res+32)*(4-n_lesion))) * 255], axis=1)

    return img_row


def pad_img(img, patch_size, stride):
    h, w = img.shape

    desired_h = math.ceil( (h - patch_size) / stride ) * stride + patch_size
    desired_w = math.ceil( (w - patch_size) / stride ) * stride + patch_size

    delta_w = desired_w - w
    delta_h = desired_h - h

    top, bottom = 0, delta_h
    left, right = 0, delta_w

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def crop_img(img, patch_size, stride, view):
    h, w = img.shape

    desired_h = math.ceil( (h - patch_size) / stride ) * stride
    desired_w = math.ceil( (w - patch_size) / stride ) * stride

    if view == 'L':
        img = img[:desired_h, :desired_w]
    else:
        img = img[:desired_h, w - desired_w:]

    return img


def whole_image_inference(fname, data_res, model, transform):
    cache_dir = os.path.dirname(fname).replace('../patch/version_0/test_data/', './cache/')
    os.makedirs(cache_dir, exist_ok=True)
    view_pos = os.path.basename(cache_dir).split('_')[0]
    cropped_img = crop_img(io.imread(fname), data_res, data_res, view_pos)
    cache_f = os.path.join(cache_dir, 'lesion_patches.json')
    if os.path.exists(cache_f):
        with open(cache_f, 'r') as f:
            lesion_fs = json.load(f)
    else:
        patches = view_as_blocks(cropped_img, (data_res, data_res)).squeeze()
        mask = (np.count_nonzero(patches, axis=(2, 3)) / data_res ** 2) > 0.9
        val_inds = np.argwhere(mask > 0)
        val_patches = patches[val_inds[:, 0], val_inds[:, 1], ...]

        inputs = torch.stack([transform(Image.fromarray(i)) for i in val_patches], dim=0)
        with torch.no_grad():
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

        lesion_fs = []
        for i, p in enumerate(pred):
            if p == 1:
                print(val_inds[i])
                x_start, y_start = val_inds[i]
                lesion_f = f'{cache_dir}/lesion_{x_start}_{y_start}.png'
                im = Image.fromarray(val_patches[i])
                im.save(lesion_f)
                lesion_fs.append(lesion_f)

        with open(cache_f, 'w') as f:
            json.dump(lesion_fs, f)

    return cropped_img, lesion_fs


def inference(model, img, transform, device):
    input_img = transform(Image.fromarray(img))
    output = model(input_img.unsqueeze(1).to(device))
    prob = F.softmax(output, dim=1)
    pred = torch.argmax(prob, dim=1)

    return pred.item(), torch.max(prob[0]).item()