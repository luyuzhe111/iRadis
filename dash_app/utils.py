import math
import cv2


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