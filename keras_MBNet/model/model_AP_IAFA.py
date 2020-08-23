# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
import numpy as np
import math
from .deform_layers import ConvOffset2D

def prior_probability(probability=0.01):
	def f(shape, dtype=K.floatx()):
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		return result
	return f

def AP(input,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
    # the first layer modified from256 to 512
    # x = Convolution2D(filters, kersize, padding='same', activation='relu',
    #                   kernel_initializer='glorot_normal', name=name + '_conv', trainable=trainable)(input)

    x_class = Convolution2D(num_anchors, (3, 3),activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_classnew', padding='same', trainable=trainable)(input)
    x_class_reshape = Reshape((-1, 1), name=name+'_class_reshapenew')(x_class)

    x_regr = Convolution2D(num_anchors * 4, (3, 3), activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regressnew', padding='same', trainable=trainable)(input)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshapenew')(x_regr)
    return x_class_reshape, x_regr_reshape

def IAFA(input,input_rgb,input_lwir,illuminate,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
    tf_score_onesub = Lambda(lambda x: 1 - x)
    illuminate_rgb_weight =illuminate
    illuminate_lwir_weight = tf_score_onesub(illuminate)

    x = Convolution2D(96, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv_all_new', trainable=trainable)(input)
    x_rgb = Convolution2D(48, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv_rgb_new', trainable=trainable)(input_rgb)
    x_lwir = Convolution2D(48, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv_lwir_new', trainable=trainable)(input_lwir)

    # modality alignment
    x = ConvOffset2D(96, name=name + 'deformconv_offset')(x)
    x_rgb = ConvOffset2D(48, name=name + 'deformconv_offset_rgb')(x_rgb)
    x_lwir = ConvOffset2D(48, name=name + 'deformconv_offset_lwir')(x_lwir)

    x_rgb_c =  Concatenate()([x,x_rgb])
    x_lwir_c =  Concatenate()([x,x_lwir])
    x_c = Concatenate()([x,x_rgb,x_lwir])

    x_regr = Convolution2D(num_anchors * 4, (3, 3),padding='same', activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regress_new',trainable=trainable)(x_c)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshape')(x_regr)


    x_class_rgb = Convolution2D(num_anchors, (3, 3), padding='same',activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_class_rgb',trainable=trainable)(x_rgb_c)
    x_class_reshape_rgb = Reshape((-1, 1), name=name+'_class_reshape_rgb')(x_class_rgb)

    x_class_lwir = Convolution2D(num_anchors, (3, 3),padding='same',activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_class_lwir',trainable=trainable)(x_lwir_c)
    x_class_reshape_lwir = Reshape((-1, 1), name=name+'_class_reshape_lwir')(x_class_lwir)


    cls_rgb_illuminate = multiply([x_class_reshape_rgb, illuminate_rgb_weight])
    cls_lwir_illuminate = multiply([x_class_reshape_lwir, illuminate_lwir_weight])
    x_class_reshape = Add()([cls_rgb_illuminate, cls_lwir_illuminate])

    return x_class_reshape, x_regr_reshape

def AP_stage(AP_CONV,base_layers,num_anchors,filters=256,kersize=(3,3), trainable=True):
    # stage3 = base_layers[0]
    # stage4 = base_layers[3]
    # stage5 = base_layers[6]
    # stage6 = base_layers[9]
    stage3 = AP_CONV[0]
    stage4 = AP_CONV[1]
    stage5 = AP_CONV[2]
    stage6 = AP_CONV[3]

    P3_cls, P3_regr  = AP(stage3, num_anchors[0], name='pred0_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = AP(stage4, num_anchors[1], name='pred1_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr  = AP(stage5, num_anchors[2], name='pred2_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = AP(stage6, num_anchors[3], name='pred3_1_base', filters=filters, kersize=kersize, trainable=trainable)
    y_cls = Concatenate(axis=1, name='mbox_cls_1')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_1')([P3_regr, P4_regr, P5_regr, P6_regr])
    return [y_cls, y_regr]

def IAFA_stage(AP_CONV,base_layers, num_anchors, filters=256, kersize=(3,3),trainable=True):
    # stage3 = base_layers[0]
    stage3 = AP_CONV[0]
    stage3_rgb = base_layers[1]
    stage3_lwir = base_layers[2]
    # stage4 = base_layers[3]
    stage4 = AP_CONV[1]
    stage4_rgb = base_layers[4]
    stage4_lwir = base_layers[5]
    # stage5 = base_layers[6]
    stage5 =  AP_CONV[2]
    stage5_rgb = base_layers[7]
    stage5_lwir = base_layers[8]
    # stage6 = base_layers[9]
    stage6 = AP_CONV[3]
    stage6_rgb = base_layers[10]
    stage6_lwir = base_layers[11]
    illuminate = base_layers[12]

    P3_cls, P3_regr = IAFA(stage3,stage3_rgb,stage3_lwir,illuminate,num_anchors[0], name='pred0_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = IAFA(stage4,stage4_rgb,stage4_lwir,illuminate,num_anchors[1], name='pred1_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = IAFA(stage5,stage5_rgb,stage5_lwir,illuminate,num_anchors[2], name='pred2_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = IAFA(stage6,stage6_rgb,stage6_lwir,illuminate,num_anchors[3], name='pred3_2_base', filters=filters, kersize=kersize, trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls_2')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_2')([P3_regr, P4_regr, P5_regr, P6_regr])
    return [y_cls, y_regr]


def create_AP_IAFA(base_layers, num_anchors,trainable=True):
    stage3 = base_layers[0]
    stage4 = base_layers[3]
    stage5 = base_layers[6]
    stage6 = base_layers[9]

    stage3 = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage3_conv_new', trainable=trainable)(stage3)
    stage4 = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage4_conv_new', trainable=trainable)(stage4)
    stage5 = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage5_conv_new', trainable=trainable)(stage5)
    stage6 = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage6_conv_new', trainable=trainable)(stage6)
    AP_CONV = [stage3,stage4,stage5,stage6]
    AP_predict = AP_stage(AP_CONV,base_layers, num_anchors, trainable=trainable)
    IAFA_predict = IAFA_stage(AP_CONV,base_layers, num_anchors, trainable=trainable)
    return AP_predict, IAFA_predict

