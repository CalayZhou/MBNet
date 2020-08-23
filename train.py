from __future__ import division
import random
import os
from keras_MBNet import config
import numpy as np

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

C.init_lr = 1e-4
C.alpha = 0.999  #0.999 update rate of weight moving average

# define the path for loading the initialized weight files
if C.network=='resnet50':
    weight_path = './data/models/double_resnet.hdf5'
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define output path for weight files
out_path = 'output/valmodels/%s/%dstep/%s' % (C.network, C.steps, C.init_lr)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the training data
kaist_numpy=np.load('./data/cache/kaist_train_data.npy',encoding="latin1",allow_pickle=True)

train_data=kaist_numpy.tolist()
num_imgs_train = len(train_data)
random.shuffle(train_data)
print('num of training samples: ',num_imgs_train)

# start training
C.neg_overlap_step2 = 0.5
C.pos_overlap_step2 = 0.7
from keras_MBNet.model.MBNetModel import MBNetModel
model = MBNetModel()
model.initialize(C)


model.creat_MBNet_model(C, train_data, phase='train')
model.train_MBNet(C, weight_path, out_path)
