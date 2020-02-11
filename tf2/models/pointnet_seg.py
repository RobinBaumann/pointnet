from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import Input, Lambda, concatenate
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import dense_bn, conv1d_bn
import pointnet_base

def get_model(input_shape, classes):
    """
    PointNet model for segmentation
    :param input_shape: shape of the input point clouds (NxK)
    :param classes: number of classes in the segmentation problem
    :param activation: activation of the last layer
    :return: Keras model of the classification network
    """

    assert K.image_data_format() == 'channels_last'

    inputs = Input(input_shape, name='Input_cloud')
    net, local_features = pointnet_base.get_model(inputs)

    global_feature_expanded = Lambda(K.expand_dims, arguments={'axis': 1})(net)
    global_feature_tiled = Lambda(K.tile, arguments={'n': [1, K.shape(local_features)[1], 1]})(global_feature_expanded)

    net = Lambda(concatenate)([local_features, global_feature_tiled])

    net = conv1d_bn(net, num_filters=512, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv1')
    net = conv1d_bn(net, num_filters=256, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv2')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv3')

    point_features = net
    net = conv1d_bn(point_features, num_filters=128, kernel_size=1, padding='valid', scope='seg_conv4', activation='softmax')
    net = conv1d_bn(net, num_filters=len(classes), kernel_size=1, padding='valid', activation='softmax')

    model = Model(inputs=inputs, outputs=net, name='pointnet_seg')

    return model