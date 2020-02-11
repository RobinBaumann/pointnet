from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import Input, Dropout, Dense
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import dense_bn, conv1d_bn
import pointnet_base


def get_model(input_shape, classes, activation=None):
    """
    PointNet model for object classification
    :param input_shape: shape of the input point clouds (NxK)
    :param classes: number of classes in the classification problem; if dict, construct multiple disjoint top layers
    :param activation: activation of the last layer
    :return: Keras model of the classification network
    """

    assert K.image_data_format() == 'channels_last'
    # Generate input tensor and get base network
    inputs = Input(input_shape, name='Input_cloud')
    maxpool, _ = pointnet_base.get_model(inputs)

    # Top layers
    if isinstance(classes, dict):
        # Fully connected layers
        net = [dense_bn(maxpool, units=512, scope=r + '_fc1', activation='relu') for r in classes]
        net = [Dropout(0.3, name=r + '_dp1')(n) for r, n in zip(classes, net)]
        net = [dense_bn(n, units=256, scope=r + '_fc2', activation='relu') for r, n in zip(classes, net)]
        net = [Dropout(0.3, name=r + '_dp2')(n) for r, n in zip(classes, net)]
        net = [Dense(units=classes[r], activation=activation, name=r)(n) for r, n in zip(classes, net)]
    else:
        net = dense_bn(maxpool, units=512, scope='fc1', activation='relu')
        net = Dropout(0.3, name='dp1')(net)
        net = dense_bn(net, units=256, scope='fc2', activation='relu')
        net = Dropout(0.3, name='dp2')(net)
        net = Dense(units=classes, name='fc3', activation=activation)(net)

    model = Model(inputs, net, name='pointnet_cls')

    return model