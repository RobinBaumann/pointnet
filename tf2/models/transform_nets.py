import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.layers import Dense, Reshape, GlobalMaxPooling1D
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import conv1d_bn, dense_bn, OrthogonalRegularizer


def transform_net(inputs, scope=None, regularize=False):
    """
    Generates an orthogonal transformation tensor for the input preprocessing
    :param inputs: tensor with input image (either BxNxK or BxNx1xK)
    :param scope: name of the grouping scope
    :param regularize: enforce orthogonality constraint
    :return: BxKxK tensor of the transformation
    """
    with K.name_scope(scope):

        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]

        net = conv1d_bn(inputs, num_filters=64, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv3')

        net = GlobalMaxPooling1D(data_format='channels_last')(net)

        net = dense_bn(net, units=512, scope='tfc1', activation='relu')
        net = dense_bn(net, units=256, scope='tfc2', activation='relu')

        transform = Dense(units=k * k,
                          kernel_initializer='zeros', bias_initializer=Constant(np.eye(k).flatten()),
                          activity_regularizer=None)(net) # OrthogonalRegularizer(l2=0.001) if regularize else
        transform = Reshape((k, k))(transform)

    return transform