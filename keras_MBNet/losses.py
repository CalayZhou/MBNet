# -*- coding:UTF-8 -*-
from keras import backend as K

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

epsilon = 1e-4
def illumination_loss(y_true, y_pred):
	y_pred = tf.reshape(y_pred, [-1, 2])
	y_true = tf.to_int32(tf.reshape(y_true, [-1])) #python27
	# y_true = tf.cast(tf.reshape(y_true, [-1]))  #python36
	illumination_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred ))
	return illumination_loss

def regr_loss(y_true, y_pred):
	positives = y_true[:, :, 0]
	n_positive = tf.reduce_sum(positives)
	absolute_loss = tf.abs(y_true[:,:,1:] - y_pred)
	square_loss = 0.5 * (y_true[:,:,1:] - y_pred) ** 2
	l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
	localization_loss = tf.to_float(tf.reduce_sum(l1_loss, axis=-1))
	loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)/ tf.maximum(1.0, n_positive)
	return loc_loss




def cls_loss(y_true, y_pred):

	y_true = tf.cast(y_true, tf.float32)
	positives = y_true[:, :, 0]
	negatives = y_true[:, :, 1]
	valid = positives + negatives

	logit = tf.clip_by_value(y_pred[:, :, 0], 1e-10, 1.0 -  1e-10)
	output = tf.log(logit / (1 - logit))
	cross_entropy = -positives*tf.log(tf.sigmoid(output)) - (1.0 - positives)*tf.log((1.0-tf.sigmoid(output)))
	classification_loss = valid * cross_entropy


	# return classification_loss
	# firstly compute the focal weight
	foreground_alpha = positives * tf.constant(0.25)
	background_alpha = negatives * tf.constant(0.75)
	foreground_weight = foreground_alpha * (tf.constant(1.0) - y_pred[:, :, 0]) ** tf.constant(2.0)
	background_weight = background_alpha * y_pred[:, :, 0] ** tf.constant(2.0)
	focal_weight = foreground_weight + background_weight
	assigned_boxes = tf.reduce_sum(positives)
	class_loss = tf.reduce_sum(classification_loss * focal_weight, axis=-1) / tf.maximum(1.0, assigned_boxes)
	#
	return class_loss
