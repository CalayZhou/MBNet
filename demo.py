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
out_path = './data/kaist_demo/'
val_data = ['set08_V000_I02159']

#initialize model
model = MBNetModel()
model.creat_MBNet_model(C, val_data, phase='inference')

#test model
model.demo_MBNet(C, val_data, weight_path,out_path)