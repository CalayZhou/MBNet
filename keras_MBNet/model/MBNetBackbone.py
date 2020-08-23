# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
from .FixedBatchNormalization import FixedBatchNormalization
import numpy as np
import tensorflow as tf
from keras.layers import Multiply, multiply
from .keras_layer_L2Normalization import L2Normalization
from .scale_bias import Scale_bias


def identity_block(input_tensor, input_tensor_mix,kernel_size, filters, stage, block, dila=(1, 1), modality ='',noadd = True,trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a'+modality, trainable=trainable)(input_tensor_mix)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a'+modality)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same',
                      name=conv_name_base + '2b'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b'+modality)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c'+modality)(x)
    #add in the DMAF block
    if noadd:
        return x,input_tensor
    #normal
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, input_tensor_mix,kernel_size, filters, stage, block, strides=(2, 2), dila=(1, 1), modality ='',noadd = True,trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a'+modality, trainable=trainable)(
        input_tensor_mix)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a'+modality)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same',
                      name=conv_name_base + '2b'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b'+modality)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c'+modality)(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1'+modality, trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1'+modality)(shortcut)
    #add in the DMAF block
    if noadd:
        return x,shortcut
    #normal
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


weight_average = Lambda(lambda x: x * 0.5)
weight_multiply = Lambda(lambda x: x[0] * x[1])
weight_div = Lambda(lambda x: tf.div(x[0] , x[1]))
subtract_feature = Lambda(lambda x:tf.subtract(x[0],x[1]))


# DMAF module
def DM_aware_fusion(x, x_lwir):
    subtracted = subtract_feature([x, x_lwir])
    subtracted_weight = GlobalAveragePooling2D()(subtracted)
    excitation_weight = Activation('tanh')(subtracted_weight)

    subtracted2 = subtract_feature([x_lwir, x])
    subtracted_weight2 = GlobalAveragePooling2D()(subtracted2)
    excitation_weight2 = Activation('tanh')(subtracted_weight2)

    x_weight=multiply([x,excitation_weight])
    x_lwir_weight=multiply([x_lwir,excitation_weight2])

    x_mix = Add()([x_lwir_weight, x])
    x_lwir_mix = Add()([x_lwir, x_weight])
    return x_mix,x_lwir_mix

#illumination mechanism
def illumination_mechanism(input_tensor_rgb,trainable=True):
    if K.image_dim_ordering() == 'tf':#bn_axis = 3
        bn_axis = 3
    else:
        bn_axis = 1
    # normalize the imput
    tf_resize_images = Lambda(lambda x: tf.image.resize_bilinear(x, [56, 56]))
    tf_div = Lambda(lambda x: x / 255)
    tf_original1 = Lambda(lambda x: x[:, :, :, 0] + 103.939)
    tf_original2 = Lambda(lambda x: x[:, :, :, 1] + 116.779)
    tf_original3 = Lambda(lambda x: x[:, :, :, 2] + 123.68)
    tf_expand_dims = Lambda(lambda x: tf.expand_dims(x, -1))

    img_input_rgb1 = tf_original1(input_tensor_rgb)
    img_input_rgb1 = tf_expand_dims(img_input_rgb1)
    img_input_rgb2 = tf_original2(input_tensor_rgb)
    img_input_rgb2 = tf_expand_dims(img_input_rgb2)
    img_input_rgb3 = tf_original3(input_tensor_rgb)
    img_input_rgb3 = tf_expand_dims(img_input_rgb3)
    img_input_rgb_pre = Concatenate()([img_input_rgb1, img_input_rgb2, img_input_rgb3])

    img_input_concat_resize = tf_resize_images(img_input_rgb_pre)
    img_input_concat_resize = tf_div(img_input_concat_resize)

    # predict the w_n,w_d
    img_input_concat_stage1 = Convolution2D(64, (3, 3), strides=(1, 1), name='illuminate_aware_stage1', padding='same',kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_resize)
    img_input_concat_stage1 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage1_bn')(img_input_concat_stage1)
    img_input_concat_stage1 = Activation('relu')(img_input_concat_stage1)
    img_input_concat_stage1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage1)

    img_input_concat_stage2 = Convolution2D(32, (3, 3), strides=(1, 1), name='illuminate_aware_stage2', padding='same',kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_stage1)
    img_input_concat_stage2 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage2_bn')(img_input_concat_stage2)
    img_input_concat_stage2 = Activation('relu')(img_input_concat_stage2)
    img_input_concat_stage2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage2)
    img_input_concat_stage2 = Flatten()(img_input_concat_stage2)

    img_input_concat_dense = Dense(units=128, activation='relu', name='illuminate_aware_dense1')(img_input_concat_stage2)
    img_input_concat_dense = Dropout(0.5)(img_input_concat_dense)
    img_input_concat_dense = Dense(units=64, activation='relu', name='illuminate_aware_dense2')(img_input_concat_dense)
    img_input_concat_dense = Dropout(0.5)(img_input_concat_dense)
    w_n = Dense(units=1, activation='relu', name='illuminate_aware_dense3')(img_input_concat_dense)
    w_d = Dense(units=1, activation='relu', name='illuminate_aware_dense4')(img_input_concat_dense)
    illuminate_output = Concatenate()([w_n, w_d])

    w_n_weight = Activation('sigmoid')(w_n)  # LWIR
    w_d_weight = Activation('sigmoid')(w_d)  # RGB

    # predict the w_absolute(|w|)
    img_input_concat_stage22 = Convolution2D(32, (3, 3), strides=(1, 1), name='illuminate_aware_stage22',padding='same', kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_stage1)
    img_input_concat_stage22 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage22_bn')(img_input_concat_stage22)
    img_input_concat_stage22 = Activation('relu')(img_input_concat_stage22)
    img_input_concat_stage22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage22)
    img_input_concat_stage22 = Flatten()(img_input_concat_stage22)

    img_input_concat_dense_alf = Dense(units=128, activation='sigmoid', name='illuminate_aware_dense1_alf')(img_input_concat_stage22)
    img_input_concat_dense_alf = Dropout(0.5)(img_input_concat_dense_alf)
    img_input_concat_dense_alf = Dense(units=64, activation='sigmoid', name='illuminate_aware_dense2_alf')(img_input_concat_dense_alf)
    img_input_concat_dense_alf = Dropout(0.5)(img_input_concat_dense_alf)
    w_absolute = Dense(units=1, activation='sigmoid', name='illuminate_aware_dense3_alf')(img_input_concat_dense_alf)

    illuminate_aware_alf_value = Scale_bias(gamma_init=1.0, beta_init=0.0,name='illuminate_aware_alf_value_scale_bais')(w_absolute)

    tf_half_add = Lambda(lambda x: 0.5 + x)
    tf_sub = Lambda(lambda x: (x[0] - x[1])*0.5)
    # the final illumination weight
    w_n_illuminate = Activation('tanh')(w_n)  # LWIR
    w_d_illuminate = Activation('tanh')(w_d)  # RGB
    illuminate_rgb_positive = tf_sub([w_d_illuminate,w_n_illuminate])
    # illuminate_rgb_positive = tf_half_sub(w_n_illuminate)
    illuminate_aware_alf_pre = multiply([illuminate_rgb_positive, illuminate_aware_alf_value])
    w_rgb = tf_half_add(illuminate_aware_alf_pre)
    return illuminate_output,w_n_weight,w_d_weight,w_rgb


#illumination Gate
def Illumination_Gate(stage_rgb,stage_lwir,channel_num ,bn_axis,w_d_weight,w_n_weight,stage_name = 'stage3',trainable=True):
    stage_rgb = multiply([stage_rgb,w_d_weight])
    stage_lwir = multiply([stage_lwir,w_n_weight])
    stage_concat=Concatenate()([stage_rgb,stage_lwir])
    stage_concat = L2Normalization(gamma_init=10, name=stage_name+'_cat_bn_pre')(stage_concat)
    stage = Convolution2D(channel_num, (1, 1), strides=(1, 1),name=stage_name+'_concat_new',padding='same',kernel_initializer='glorot_normal', trainable=trainable)(stage_concat)
    stage = FixedBatchNormalization(axis=bn_axis, name=stage_name+'_cat_bn_new')(stage)
    stage = Activation('relu')(stage)
    return stage,stage_rgb,stage_lwir

def ResNet_DMAF_Block(x,x_lwir,x_mix,x_lwir_mix, stage_order =3,identity_block_num = 0,channel_size = 128, trainable = False):
    block_order=['a','b','c','d','e','f']

    x, x_shortcut = conv_block(x, x_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block='a', strides=(2, 2),
                               modality='', noadd=True, trainable=trainable)
    x_lwir, x_lwir_shortcut = conv_block(x_lwir, x_lwir_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block='a',
                                         strides=(2, 2), modality='_lwir', noadd=True, trainable=trainable)
    x_pre = Add()([x, x_shortcut])
    x_lwir_pre = Add()([x_lwir, x_lwir_shortcut])
    x = Activation('relu')(x_pre)
    x_lwir = Activation('relu')(x_lwir_pre)
    if identity_block_num==0:
        return x, x_lwir
    x_mix, x_lwir_mix = DM_aware_fusion(x, x_lwir)

    for identity_i in range(identity_block_num):
        x,x_input_tensor = identity_block(x,x_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block=block_order[identity_i+1], modality='',noadd = True,trainable = trainable)
        x_lwir,x_lwir_input_tensor = identity_block(x_lwir, x_lwir_mix,3, [channel_size, channel_size, channel_size*4],stage=stage_order, block=block_order[identity_i+1],modality='_lwir',noadd = True, trainable=trainable)
        x_pre = Add()([x, x_input_tensor])
        x_lwir_pre = Add()([x_lwir,x_lwir_input_tensor])
        x = Activation('relu')(x_pre)
        x_lwir = Activation('relu')(x_lwir_pre)
        x_mix, x_lwir_mix = DM_aware_fusion(x, x_lwir)

    return  x,x_lwir,x_mix,x_lwir_mix



def MBNetBackbone(input_tensor_rgb=None, input_tensor_lwir=None,trainable=False):
    img_input_rgb = input_tensor_rgb
    img_input_lwir = input_tensor_lwir
    if K.image_dim_ordering() == 'tf':#bn_axis = 3
        bn_axis = 3
    else:
        bn_axis = 1
    # illumination mechanism
    # w_n_weight, w_d_weight->used in illumination gate; illuminate_output->used for illumination loss; w_rgb->used for IAFA module
    illuminate_output, w_n_weight, w_d_weight, w_rgb = illumination_mechanism(input_tensor_rgb,trainable= trainable)

    print('Froze the first two stage layers')
    x = ZeroPadding2D((3, 3))(img_input_rgb)
    x_lwir = ZeroPadding2D((3, 3))(img_input_lwir)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = False)(x)
    x_lwir = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1_lwir', trainable=False)(x_lwir)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x_lwir = FixedBatchNormalization(axis=bn_axis, name='bn_conv1_lwir')(x_lwir)
    x = Activation('relu')(x)
    x_lwir = Activation('relu')(x_lwir)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x_lwir = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_lwir)
    x = conv_block(x, x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),modality='', noadd = False,trainable = False)
    x_lwir = conv_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), modality='_lwir',noadd = False,trainable=False)
    x = identity_block(x, x, 3, [64, 64, 256], stage=2, block='b', modality='', noadd = False,trainable =False )
    x_lwir = identity_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='b',  modality='_lwir',noadd = False,trainable=False)
    stage2_rgb = identity_block(x, x, 3, [64, 64, 256], stage=2, block='c',modality='',noadd = False,  trainable = False)
    stage2_lwir = identity_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='c',  modality='_lwir',noadd = False,trainable=False)

    print('the MBNet backbone (ResNet50 embeded with DMAF module)')
    stage3_rgb,stage3_lwir,stage3_rgb_mix,stage3_lwir_mix = ResNet_DMAF_Block(stage2_rgb, stage2_lwir, stage2_rgb, stage2_lwir,stage_order=3, identity_block_num=3, channel_size=128, trainable=trainable)
    stage4_rgb,stage4_lwir,stage4_rgb_mix,stage4_lwir_mix = ResNet_DMAF_Block(stage3_rgb, stage3_lwir,stage3_rgb_mix,stage3_lwir_mix,stage_order=4,identity_block_num=5, channel_size=256,trainable=trainable)
    stage5_rgb,stage5_lwir,stage5_rgb_mix,stage5_lwir_mix = ResNet_DMAF_Block(stage4_rgb,stage4_lwir,stage4_rgb_mix,stage4_lwir_mix,stage_order=5,identity_block_num=2, channel_size=512,trainable=trainable)
    stage6_rgb,stage6_lwir = ResNet_DMAF_Block(stage5_rgb,stage5_lwir,stage5_rgb_mix,stage5_lwir_mix,stage_order=6,identity_block_num=0, channel_size=256,trainable=trainable)
    print('the illumination gate')
    stage3,stage3_rgb,stage3_lwir = Illumination_Gate(stage3_rgb, stage3_lwir, 512, bn_axis, w_d_weight, w_n_weight, stage_name='stage3',trainable=trainable)
    stage4,stage4_rgb,stage4_lwir = Illumination_Gate(stage4_rgb, stage4_lwir, 1024, bn_axis, w_d_weight, w_n_weight, stage_name='stage4',trainable=trainable)
    stage5,stage5_rgb,stage5_lwir = Illumination_Gate(stage5_rgb, stage5_lwir, 2048, bn_axis, w_d_weight, w_n_weight, stage_name='stage5',trainable=trainable)
    stage6,stage6_rgb,stage6_lwir = Illumination_Gate(stage6_rgb, stage6_lwir, 2048, bn_axis, w_d_weight, w_n_weight, stage_name='stage6',trainable=trainable)

    predictor_sizes = np.array([stage3._keras_shape[1:3],
                                stage4._keras_shape[1:3],
                                stage5._keras_shape[1:3],
                                np.ceil(np.array(stage5._keras_shape[1:3]) / 2)])

    return [stage3,stage3_rgb,stage3_lwir,  stage4,stage4_rgb,stage4_lwir,
            stage5,stage5_rgb,stage5_lwir,  stage6,stage6_rgb,stage6_lwir,  w_rgb],\
           [illuminate_output], predictor_sizes

