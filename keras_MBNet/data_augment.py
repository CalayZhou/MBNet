# -*- coding:UTF-8 -*-
from __future__ import division
import cv2
import numpy as np
import copy


def _hue_kaist(image,image_lwir, min=0.75, max=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    # x1=hsv[:,:,0]#0-179
    random_br = np.random.uniform(min, max)
    mask = hsv[:,:,0] * random_br > 179
    mask_lwir = hsv_lwir[:,:,0] * random_br> 179
    v_channel = np.where(mask, 179, hsv[:,:,0] * random_br)
    v_channel_lwir = np.where(mask_lwir, 179, hsv_lwir[:,:,0] * random_br)
    hsv[:,:,0] = v_channel
    hsv_lwir[:,:,0] = v_channel_lwir

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.cvtColor(hsv_lwir, cv2.COLOR_HSV2RGB)

def _saturation_kaist(image,image_lwir, min=0.75, max=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    # x1 = hsv[:, :, 1]#0-255
    random_br = np.random.uniform(min, max)
    mask = hsv[:,:,1] * random_br > 255
    mask_lwir = hsv_lwir[:,:,1] * random_br> 255
    v_channel = np.where(mask, 255, hsv[:,:,1] * random_br)
    v_channel_lwir = np.where(mask_lwir, 255, hsv_lwir[:,:,1] * random_br)
    hsv[:,:,1] = v_channel
    hsv_lwir[:,:,1] = v_channel_lwir

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.cvtColor(hsv_lwir, cv2.COLOR_HSV2RGB)

def _brightness_kaist(image,image_lwir, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min,max)
    mask = hsv[:,:,2] * random_br > 255
    mask_lwir = hsv_lwir[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    v_channel_lwir = np.where(mask_lwir, 255, hsv_lwir[:,:,2] * random_br)
    hsv[:,:,2] = v_channel
    hsv_lwir[:,:,2] = v_channel_lwir
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),cv2.cvtColor(hsv_lwir,cv2.COLOR_HSV2RGB)

def augment_lwir(img_data, c):

    assert 'filepath_lwir' in img_data
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img_lwir = cv2.imread('./'+img_data_aug['filepath_lwir'])
    img = cv2.imread('./'+img_data_aug['filepath'])
    img_height, img_width = img.shape[:2]
    # random brightness
    if c.brightness and np.random.randint(0, 2) == 0:
        img,img_lwir = _brightness_kaist(img,img_lwir, min=c.brightness[0], max=c.brightness[1])
    #5.10
    # random hue
    if  np.random.randint(0, 2) == 0:
        img, img_lwir = _hue_kaist(img, img_lwir)
    # random saturation
    if np.random.randint(0, 2) == 0:
        img, img_lwir = _saturation_kaist(img, img_lwir)
    # random horizontal flip
    if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        img_lwir = cv2.flip(img_lwir, 1)

        if len(img_data_aug['bboxes']) > 0:
            img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
    # random crop a patch
    ratio = np.random.uniform(c.scale[0], c.scale[1])
    crop_h, crop_w = np.asarray(ratio * np.asarray(img.shape[:2]), dtype=np.int)
    gts = np.copy(img_data_aug['bboxes'])
    if len(gts) > 0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h) + crop_h * 0.5)
    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))#如果x1 y1比0小置0
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    patch_X = np.copy(img[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    patch_X_lwir = np.copy(img_lwir[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])

    img = patch_X
    img_lwir=patch_X_lwir

    if len(gts) > 0:
        before_limiting = copy.deepcopy(gts)
        gts[:, [0, 2]] -= crop_x1
        gts[:, [1, 3]] -= crop_y1
        y_coords = gts[:, [1, 3]]
        y_coords[y_coords < 0] = 0
        y_coords[y_coords >= crop_h] = crop_h - 1
        gts[:, [1, 3]] = y_coords
        x_coords = gts[:, [0, 2]]
        x_coords[x_coords < 0] = 0
        x_coords[x_coords >= crop_w] = crop_w - 1
        gts[:, [0, 2]] = x_coords
        before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
            before_limiting[:, 3] - before_limiting[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        gts = gts[after_area >= c.in_thre * before_area]

    # resize to original image size
    img = cv2.resize(img, dsize=(c.random_crop[1], c.random_crop[0]))
    img_lwir = cv2.resize(img_lwir, dsize=(c.random_crop[1], c.random_crop[0]))

    reratio = crop_h/c.random_crop[0]
    if len(gts) > 0:
        gts = (gts/reratio).astype(np.int)
        w = gts[:,2]-gts[:,0]
        gts = gts[w>=16,:]

    img_data_aug['bboxes'] = gts


    img_data_aug['width'] = c.random_crop[1]
    img_data_aug['height'] = c.random_crop[0]
    return img_data_aug, img,img_lwir
