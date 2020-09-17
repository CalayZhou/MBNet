
class Config:

	def __init__(self):
		self.gpu_ids = '0'# if you have more gpus, setting a larger batch size may get better results
		self.onegpu = 10 #batch size in one gpu
		self.num_epochs = 8
		self.reduce_lr_epoch=6
		self.add_epoch = 0
		self.iter_per_epoch = 8963
		self.init_lr = 1e-4
		self.useseg=True
		self.cross_residual=True

		# setting for network architechture
		self.network = 'resnet50'
		self.steps = 2

		# setting for data augmentation
		self.use_horizontal_flips = True
		self.brightness = (0.5, 2, 0.5)
		self.in_thre = 0.5
		self.scale = (0.3, 1.0)
		self.random_crop = (512, 640)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]

		# setting for scales
		self.anchor_box_scales = [[25.84, 29.39], [33.81, 38.99], [44.47, 52.54], [65.80, 131.40]]
		self.anchor_ratios = [[0.41], [0.41], [0.41], [0.41]]

		# scaling the stdev
		self.classifier_regr_std = [0.1, 0.1, 0.2, 0.2]

		# overlaps for ignore areas
		self.ig_overlap = 0.5
		# overlaps for different ALF steps
		self.neg_overlap_step1 = 0.3
		self.pos_overlap_step1 = 0.5
		self.neg_overlap_step2 = 0.5
		self.pos_overlap_step2 = 0.7

		# setting for inference
		self.scorethre= 0.01
		self.overlap_thresh = 0.425
		self.pre_nms_topN = 6000
		self.post_nms_topN = 100
		self.roi_stride= 16

