import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import Input, Dropout, Dot, Lambda,  concatenate, GlobalMaxPooling1D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import conv1d_bn, dense_bn, OrthogonalRegularizer
from transform_nets import transform_net


def get_model(input_point_shape, input_label_shape, cat_num, part_num, batch_size, num_point):
    """ ConvNet baseline, input is BxNx3 gray image """

    input_points = Input(input_point_shape, name='Input_cloud')
    input_labels = Input(input_label_shape, name='Input_label')

    # Obtain spatial point transform from inputs and convert inputs
    ptransform = transform_net(input_points, scope='transform_net1', regularize=False)
    point_cloud_transformed = Dot(axes=(2, 1))([input_points, ptransform])

    # First block of convolutions
    out1 = conv1d_bn(point_cloud_transformed, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv1')
    out2 = conv1d_bn(out1, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv2')

    out3 = conv1d_bn(out2, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv3')

    # Obtain feature transform and apply it to the network
    ftransform = transform_net(out3, scope='transform_net2', regularize=True)
    net_transformed = Dot(axes=(2, 1))([out3, ftransform])

    # Second block of convolutions
    out4 = conv1d_bn(net_transformed, num_filters=512, kernel_size=1, padding='valid', use_bias=True, scope='conv4')
    out5 = conv1d_bn(out4, num_filters=2048, kernel_size=1, padding='valid', use_bias=True, scope='hx')

    # add Maxpooling here, because it is needed in both nets.
    out_max = GlobalMaxPooling1D(data_format='channels_last', name='maxpool')(out5)

    # segmentation network
    one_hot_label_expand = K.reshape(input_labels, [batch_size, 1, 1, cat_num])
    out_max = Lambda(concatenate)(axis=3, values=[out_max, one_hot_label_expand])

    expand = Lambda(K.expand_dims, arguments={'axis': 1})(out_max)
    max_pool_tiled = Lambda(K.tile, arguments={'n': [1, num_point, 1]})(expand)

    concat = Lambda(concatenate)([out1, out2, out3, out4, out5, max_pool_tiled])

    net = conv1d_bn(concat, num_filters=256, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv1')
    net = Dropout(0.2, name="seg_drop1")(net)
    net = conv1d_bn(net, num_filters=256, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv2')
    net = Dropout(0.2, name="seg_drop2")(net)
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv3')

    net = conv1d_bn(net, num_filters=part_num, kernel_size=1, padding='valid', activation=None)

    model = Model(inputs=[input_points, input_labels], outputs=net, name='pointnet_part_seg')

    return model
