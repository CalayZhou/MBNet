'''
A custom Keras layer to perform L2-normalization.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np

class Scale_bias(Layer):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

    Arguments:
        gamma_init (int): The initial scaling parameter. Defaults to 20 following the
            SSD paper.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Returns:
        The scaled tensor. Same shape as the input tensor.

    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''

    def __init__(self, gamma_init=1,beta_init=0.0, **kwargs):
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        super(Scale_bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.input_spec = [InputSpec(shape=input_shape)]
        # gamma = self.gamma_init * np.ones(1)
        # beta = self.beta_init * np.ones(1)
        gamma = self.gamma_init
        beta = self.beta_init
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.beta = K.variable(beta, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma,self.beta]
        super(Scale_bias, self).build(input_shape)

    def call(self, x, mask=None):
        # output = K.l2_normalize(x, self.axis)
        # x_pre = K.dot(x, self.gamma)
        # output = K.bias_add(x_pre, self.beta)
        x_pre = x*self.gamma
        output = x_pre+self.beta

        return output
