# -*- coding:UTF-8 -*-
from __future__ import division
import os
from keras_MBNet import config
from keras_MBNet.model.MBNetModel import MBNetModel

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C.random_crop = (512, 640)
C.network = 'resnet50'

#load model weight

weight_path = './data/models/resnet_e7_l224.hdf5'
# weight_path = './output/valmodels/resnet50/2step/0.0001/resnet_e7_l224.hdf5'
data_path = './data/kaist_test/kaist_test_visible'
val_data = os.listdir(data_path)

#initialize model
model = MBNetModel()
model.creat_MBNet_model(C, val_data, phase='inference')

#test model
model.test_MBNet(C,data_path, val_data, weight_path)