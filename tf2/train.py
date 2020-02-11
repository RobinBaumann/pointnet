import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from provider import PointCloudProvider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
CKPT_DIR = os.path.join(LOG_DIR, "ckpts")
if not os.path.exists(CKPT_DIR): os.mkdir(CKPT_DIR)

MAX_NUM_POINT = 4096
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


def get_learning_rate_schedule():
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,  # Initial learning rate
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    return learning_rate        


def random_split(samples, atFraction):
    """
    Perform the train/test split.
    """
    print("atFraction = ", atFraction)
    mask = np.random.rand(len(samples)) < atFraction
    return samples[mask], samples[~mask]


def train():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[FLAGS.gpu],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9536)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    model = MODEL.get_model((None, 3), NUM_CLASSES)

    learning_rate = get_learning_rate_schedule()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # initialize Dataset
    PointCloudProvider.initialize_dataset()

    generator_training = PointCloudProvider('train', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)
    generator_validation = PointCloudProvider('test', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CKPT_DIR, save_weights_only=False, save_best_only=True),
    ]
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit_generator(generator=generator_training, validation_data=generator_validation,
                        steps_per_epoch=len(generator_training),
                        validation_steps=len(generator_validation),
                        epochs=MAX_EPOCH, callbacks=callbacks, use_multiprocessing=False)
    model.save("trained_model")


if __name__ == "__main__":
    train()
