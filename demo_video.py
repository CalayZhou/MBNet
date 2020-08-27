# -*- coding:UTF-8 -*-
from __future__ import division
import os
from keras_MBNet import config
from keras_MBNet.model.MBNetModel import MBNetModel
import cv2

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C.random_crop = (512, 640)
C.network = 'resnet50'

#load model weight

weight_path = './data/models/resnet_e7_l224.hdf5'  #weight file path
# weight_path = './output/valmodels/resnet50/2step/0.0001/resnet_e7_ l224.hdf5'

test_file = './data/kaist_demo/video/visible.mp4'  #input visible video
lwir_test_file ='./data/kaist_demo/video/lwir.mp4' #input lwir video

val_data = ['set08_V000_I02159']

#initialize model
model = MBNetModel()
model.creat_MBNet_model(C, val_data, phase='inference')


#test model
model.demo_video_MBNet(C, test_file, lwir_test_file, weight_path)

