"""
Created by Robin Baumann <robin.baumann@inovex.de> at 30.12.19.
"""

import os
import sys

from tensorflow.keras.layers import Dot, GlobalMaxPooling1D

from transform_nets import transform_net
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import conv1d_bn, dense_bn

def get_model(inputs):
    """
    Convolutional portion of model, common across different tasks (classification, segmentation, etc)
    :param inputs: Input tensor with the point cloud shape (BxNxK)
    :return: tensor layer for CONV5 activations, tensor layer with local features
    """

    # Obtain spatial point transform from inputs and convert inputs
    ptransform = transform_net(inputs, scope='transform_net1', regularize=False)
    point_cloud_transformed = Dot(axes=(2, 1))([inputs, ptransform])

    # First block of convolutions
    net = conv1d_bn(point_cloud_transformed, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv2')

    # Obtain feature transform and apply it to the network
    ftransform = transform_net(net, scope='transform_net2', regularize=True)
    net_transformed = Dot(axes=(2, 1))([net, ftransform])

    # Second block of convolutions
    net = conv1d_bn(net_transformed, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid', use_bias=True, scope='conv4')
    hx = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid', use_bias=True, scope='hx')

    # add Maxpooling here, because it is needed in both nets.
    net = GlobalMaxPooling1D(data_format='channels_last', name='maxpool')(hx)

    return net, net_transformed