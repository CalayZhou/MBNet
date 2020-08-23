# -*- coding:UTF-8 -*-
from keras.layers import Input
from keras_MBNet import data_generators
import numpy as np
from keras_MBNet import config
C = config.Config()

class Base_model():
	def name(self):
		return 'BaseModel'
	def initialize(self, opt):
		self.num_epochs = opt.num_epochs
		self.add_epoch = opt.add_epoch
		self.iter_per_epoch = opt.iter_per_epoch
		self.init_lr = opt.init_lr
		self.num_gpus = len(opt.gpu_ids.split(','))
		self.batchsize = opt.onegpu*self.num_gpus
		self.epoch_length = int(opt.iter_per_epoch / self.batchsize)
		self.total_loss_r =[]
		self.best_loss = 0

	def create_base_MBNet_model(self,opt, train_data, phase='train',wei_mov_ave = False):
		if opt.network == 'resnet50':
			from.  import MBNetBackbone as cross_nn
		else:
			raise NotImplementedError('Not support network: {}'.format(opt.network))

		#define the image input
		self.img_input_rgb = Input(shape=(opt.random_crop[0], opt.random_crop[1], 3))
		self.img_input_lwir = Input(shape=(opt.random_crop[0], opt.random_crop[1], 3))

		# define the base network
		self.base_layers, self.illuminate_output, feat_map_sizes = cross_nn.MBNetBackbone(self.img_input_rgb,self.img_input_lwir,trainable=True)
		#teacher
		self.base_layers_tea, self.illuminate_output_tea, _ = cross_nn.MBNetBackbone(self.img_input_rgb, self.img_input_lwir,trainable=True)

		# get default anchors and define data generator
		self.anchors, self.num_anchors = data_generators.get_anchors(img_height=opt.random_crop[0], img_width=opt.random_crop[1],
														   feat_map_sizes=feat_map_sizes.astype(np.int),
														   anchor_box_scales=opt.anchor_box_scales,
														   anchor_ratios=opt.anchor_ratios)
		# get the needed data for train
		if phase=='train':
			self.data_gen_train = data_generators.get_target_kaist(self.anchors, train_data, opt, batchsize=self.batchsize,
													igthre=opt.ig_overlap, posthre=opt.pos_overlap_step1,
													negthre=opt.neg_overlap_step1)



