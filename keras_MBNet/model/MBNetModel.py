# -*- coding:UTF-8 -*-
from .base_model import Base_model
from keras.optimizers import Adam
from keras.models import Model
from keras_MBNet.parallel_model import ParallelModel
from keras.utils import generic_utils
from keras_MBNet import losses as losses
from keras_MBNet import bbox_process
from . import model_AP_IAFA
import time, os, cv2
from keras_MBNet import config
C = config.Config()
from keras.layers import *

class MBNetModel(Base_model):
	def name(self):
		return 'Model_2step'
	def initialize(self, opt):
		Base_model.initialize(self,opt)
		# specify the training details
		self.cls_loss_r1 = []
		self.regr_loss_r1 = []
		self.cls_loss_r2 = []
		self.regr_loss_r2 = []
		self.illuminate_loss = []
		self.losses = np.zeros((self.epoch_length, 9))
		self.optimizer = Adam(lr=opt.init_lr)
		print ('Initializing the {}'.format(self.name()))

	def creat_MBNet_model(self,opt,train_data, phase='train'):
		Base_model.create_base_MBNet_model(self, opt,train_data, phase=phase)
		alf1, alf2 = model_AP_IAFA.create_AP_IAFA(self.base_layers,self.num_anchors, trainable=True)
		illuminate_value = self.illuminate_output
		alf1_tea, alf2_tea = model_AP_IAFA.create_AP_IAFA(self.base_layers_tea, self.num_anchors, trainable=True)
		illuminate_value_tea = self.illuminate_output_tea
		self.model_tea = Model([self.img_input_rgb,self.img_input_lwir], alf1_tea + alf2_tea+illuminate_value_tea)

		if phase=='train':
			self.model_0st = Model([self.img_input_rgb, self.img_input_lwir], illuminate_value)
			self.model_1st = Model([self.img_input_rgb,self.img_input_lwir], alf1)
			self.model_2nd = Model([self.img_input_rgb,self.img_input_lwir], alf2)
			if self.num_gpus > 1:
				self.model_1st = ParallelModel(self.model_1st, int(self.num_gpus))
				self.model_2nd = ParallelModel(self.model_2nd, int(self.num_gpus))

			self.model_0st.compile(optimizer=self.optimizer, loss=[losses.illumination_loss],sample_weight_mode=None)
			self.model_1st.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss],sample_weight_mode=None)
			self.model_2nd.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss],sample_weight_mode=None)
		self.model_all = Model([self.img_input_rgb,self.img_input_lwir], alf1+alf2+illuminate_value)



	def train_MBNet(self,opt, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		self.model_tea.load_weights(weight_path, by_name=True)
		print ('load weights from {}'.format(weight_path))
		iter_num = 0
		start_time = time.time()

		for epoch_num in range(self.num_epochs):
			progbar = generic_utils.Progbar(self.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1 + self.add_epoch, self.num_epochs + self.add_epoch))
			lr_later = K.get_value(self.model_1st.optimizer.lr)
			if  epoch_num % C.reduce_lr_epoch == 0 and epoch_num != 0 :
				lr = K.get_value(self.model_1st.optimizer.lr)
				K.set_value(self.model_1st.optimizer.lr, 0.1 *lr)
				K.set_value(self.model_2nd.optimizer.lr, 0.1 * lr)
				K.set_value(self.model_0st.optimizer.lr, 0.1 * lr)
				assert K.get_value(self.model_1st.optimizer.lr)==K.get_value(self.model_2nd.optimizer.lr)
				assert K.get_value(self.model_1st.optimizer.lr)==K.get_value(self.model_0st.optimizer.lr)
				lr_later = K.get_value(self.model_1st.optimizer.lr)
				print("model lr changed to {}".format(lr_later))

			while True:
				try:
					[X, X_lwir], Y, img_data,illumination_batch_value= next(self.data_gen_train)
					loss_s0 = self.model_0st.train_on_batch([X, X_lwir], illumination_batch_value)
					self.losses[iter_num, 4] = loss_s0

					loss_s1 = self.model_1st.train_on_batch([X, X_lwir], Y)
					self.losses[iter_num, 0] = loss_s1[1]
					self.losses[iter_num, 1] = loss_s1[2]
					pred1 = self.model_1st.predict_on_batch([X, X_lwir])
					Y2 = bbox_process.get_target_1st_posfirst(self.anchors, pred1[1], img_data, opt,
													 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
													 negthre=opt.neg_overlap_step2)
					loss_s2 = self.model_2nd.train_on_batch([X, X_lwir], Y2)
					self.losses[iter_num, 2] = loss_s2[1]
					self.losses[iter_num, 3] = loss_s2[2]
					# apply weight moving average
					for l in self.model_tea.layers:
						weights_tea = l.get_weights()
						if len(weights_tea) > 0:
							if(self.model_all.get_layer(name=l.name)==None):
								print(l.name)
							weights_stu = self.model_all.get_layer(name=l.name).get_weights()
							weights_tea = [opt.alpha * w_tea + (1 - opt.alpha) * w_stu for (w_tea, w_stu) in
										   zip(weights_tea, weights_stu)]
							l.set_weights(weights_tea)
					iter_num += 1

					if iter_num % 20 == 0:
						progbar.update(iter_num,
									   [('cls1', np.mean(self.losses[:iter_num, 0])),
										('regr1', np.mean(self.losses[:iter_num, 1])),
										('cls2', np.mean(self.losses[:iter_num, 2])),
										('regr2', np.mean(self.losses[:iter_num, 3])),
										('illuminate', np.mean(self.losses[:iter_num, 4])),
										('lr',lr_later)])


					if iter_num == (self.epoch_length // 4) or iter_num == self.epoch_length // 2 or \
							iter_num == (self.epoch_length //4)*3 or iter_num == self.epoch_length:  # 多保存几个模型
						self.model_tea.save_weights(os.path.join(out_path,'resnet_e{}_l{}.hdf5'.format(epoch_num + 1 + self.add_epoch, iter_num)))
					if iter_num == self.epoch_length:
						cls_loss1 = np.mean(self.losses[:, 0])
						regr_loss1 = np.mean(self.losses[:, 1])
						cls_loss2 = np.mean(self.losses[:, 2])
						regr_loss2 = np.mean(self.losses[:, 3])
						illuminate_loss = np.mean(self.losses[:, 4])
						total_loss = cls_loss1 + regr_loss1 + cls_loss2 + regr_loss2 + illuminate_loss

						self.total_loss_r.append(total_loss)
						self.cls_loss_r1.append(cls_loss1)
						self.regr_loss_r1.append(regr_loss1)
						self.cls_loss_r2.append(cls_loss2)
						self.regr_loss_r2.append(regr_loss2)
						self.illuminate_loss.append(np.mean(self.losses[:, 4]))

						print('Total loss: {}'.format(total_loss))
						print('Elapsed time: {}'.format(time.time() - start_time))

						iter_num = 0
						start_time = time.time()

						if total_loss < self.best_loss:
							print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, total_loss))
							self.best_loss = total_loss

						break
				except Exception as e:
					print ('Exception: {}'.format(e))
					continue
			records = np.concatenate((np.asarray(self.total_loss_r).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r1).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r1).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r2).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r2).reshape((-1, 1)),
									  np.asarray(self.illuminate_loss).reshape((-1, 1))),
									 axis=-1)
			np.savetxt(os.path.join(out_path, 'records.txt'), np.array(records), fmt='%.8f')
		print('Training complete, exiting.')

	def test_MBNet(self,opt, data_path,val_data, weight_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print ('load weights from {}'.format(weight_path))

		for f in range(len(val_data)):
			img_name = os.path.join(data_path,val_data[f])
			if not img_name.lower().endswith(('.jpg', '.png')):
				continue
			print(img_name)
			img = cv2.imread(img_name)

			img_name_lwir = os.path.join(data_path[:-7]+'lwir', val_data[f][:-11]+'lwir.png')
			print(img_name_lwir)
			img_lwir = cv2.imread(img_name_lwir)

			start_time = time.time()
			x_in = bbox_process.format_img(img, opt)
			x_in_lwir = bbox_process.format_img(img_lwir, opt)
			Y = self.model_all.predict([x_in,x_in_lwir])
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
			print ('Test time: %.4f s' % (time.time() - start_time))


			image_name_save,png=val_data[f].split('.')
			image_name_save = image_name_save[:-8]#lwir

			result_path =  './data/result'
			if not os.path.exists(result_path):
				os.makedirs(result_path)

			image_set_file = os.path.join(result_path, image_name_save + '.txt')
			list_file = open(image_set_file, 'w')
			for i in range(len(bbx)):
				image_write_txt = 'person' + ' ' + str(np.round(bbx[i][0], 4)) + ' ' + str(np.round(bbx[i][1], 4)) + ' ' \
								  + str(np.round(bbx[i][2], 4)) + ' ' + str(np.round(bbx[i][3], 4)) + ' ' + str(round(float(scores[i]), 8))
				list_file.write(image_write_txt)
				list_file.write('\n')
			list_file.close()

	def demo_MBNet(self,opt, val_data,weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print ('load weights from {}'.format(weight_path))
		for f in range(len(val_data)):
			img_name = os.path.join(out_path,val_data[f]+'_visible.png')
			if not img_name.lower().endswith(('.jpg', '.png')):
				continue
			print(img_name)
			img = cv2.imread(img_name)

			img_name_lwir = os.path.join(out_path, val_data[f]+'_lwir.png')
			print(img_name_lwir)
			img_lwir = cv2.imread(img_name_lwir)

			x_in = bbox_process.format_img(img, opt)
			x_in_lwir = bbox_process.format_img(img_lwir, opt)
			Y = self.model_all.predict([x_in,x_in_lwir])
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
			for ind in range(len(bbx)):
				if scores[ind][0]<0.5:
					continue
				(x1, y1, x2, y2) = bbx[ind,:]
				cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				#cv2.putText(img, str(scores[ind][0]), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
				cv2.rectangle(img_lwir, (x1, y1), (x2, y2), (0, 255, 0), 2)
				#cv2.putText(img_lwir, str(scores[ind][0]), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
			img_concat = np.concatenate([img,img_lwir],axis=1)
			cv2.imwrite(os.path.join(out_path, val_data[f]+'.png'),img_concat)
			
	def demo_video_MBNet(self, opt, test_file, lwir_test_file, weight_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print('loaded weights from {}'.format(weight_path))

		print('loaded visible video from :'+ test_file)

		print('loaded lwir video from :' + lwir_test_file)

		vid = cv2.VideoCapture(test_file)

		lwir_vid = cv2.VideoCapture(lwir_test_file)
		fps = vid.get(cv2.CAP_PROP_FPS)

		frame_width = int(vid.get(3))
		frame_height = int(vid.get(4))
		out_vid = cv2.VideoWriter('output_vid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
								  (frame_width, frame_height))
		out_lwir_vid = cv2.VideoWriter('output_lwir_vid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
									   (frame_width, frame_height))
		idx = 0

		while True:
			ret , frame = vid.read()

			lwir_ret, lwir_frame = lwir_vid.read()

			x_in = bbox_process.format_img(frame, opt)
			x_in_lwir = bbox_process.format_img(lwir_frame, opt)
			Y = self.model_all.predict([x_in, x_in_lwir])
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
			for ind in range(len(bbx)):
				if scores[ind][0] < 0.5:
					continue
				(x1, y1, x2, y2) = bbx[ind, :]
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				# cv2.putText(img, str(scores[ind][0]), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
				cv2.rectangle(lwir_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			out_vid.write(frame)
			out_lwir_vid.write(lwir_frame)

		vid.release()
		lwir_vid.release()
		out_vid.release()
		out_lwir_vid.release()
		cv2.destroyAllWindows()


