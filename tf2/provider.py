import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf
import os
import pandas as pd
import h5py
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


class PointCloudProvider(Sequence):
    """
    Lazily load point clouds and annotations from filesystem and prepare it for model training.
    """

    def __init__(self, mode='train', batch_size=32, n_classes=40, sample_size=1024, task="classification"):
        """
        Instantiate a data provider instance for point cloud data.
        Args:
            dataset: pandas DataFrame containing the index to the files (train or test set)
            batch_size: the desired batch size
            n_classes: The number of different classes (needed for one-hot encoding of labels)
            sample_size: the amount of points to sample per instance.
            task: string denoting the tasks for which the data is to be loaded. Either "classification" (default) or "segmentaion".
        """
        self.datasets = {
            'train': 'data/train_VDS.h5',
            'test': 'data/test_VDS.h5'
        }
        self.mode = mode
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.sample_size = sample_size
        self.task = task

        self.indices = np.arange(0,len(h5py.File(self.datasets[self.mode], 'r')['data']), 1)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil((len(self.indices)/self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        mask = np.zeros_like(self.indices).astype(np.bool)
        mask[batch_indices] = True
        X = h5py.File(self.datasets[self.mode], 'r')['data'][mask, ...]
        y = h5py.File(self.datasets[self.mode], 'r')['label'][mask, ...]

        return self.rotate_point_clouds(np.array(X)), to_categorical(np.array(y), num_classes=self.n_classes)

    def sample_random_points(self, pc):
        r_idx = np.random.randint(pc.shape[1], size=self.sample_size)
        return np.take(pc, r_idx, axis=1)

    def on_epoch_end(self):
        """Shuffle training data, so batches are in different order"""
        np.random.shuffle(self.indices)

    def rotate_point_clouds(self, batch, rotation_angle_range=(-np.pi / 8, np.pi / 8)):
        """Rotate point cloud around y-axis (=up) by random angle"""
        for b in range(batch.shape[0]):
            phi = np.random.uniform(*rotation_angle_range)
            c, s = np.cos(phi), np.sin(phi)
            R = np.asarray([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]])
            shape_pc = batch[b, ...]
            batch[b, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return batch

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def load_data_file(self, filename):
        if self.task == "classification":
            return self.load_h5(filename)
        else:
            return self.load_h5_data_label_seg(filename)

    def load_h5_data_label_seg(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        return data, label, seg

    @staticmethod
    def initialize_dataset():
        """
        Loads an index to all files and structures them.
        :param data_directory: directory containing the data files
        :param file_extension: extension of the data files
        :return: pandas dataframe containing an index to all files and a label index,
            mapping numerical label representations to label names.
        """

        print("[Provider]: Creating Virtual Dataset")

        train_index = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt')
        test_index = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt')
        train_files = [line.rstrip() for line in open(train_index)]
        test_files = [line.rstrip() for line in open(test_index)]

        def create_vds(files, prefix='train'):
            out_size = 0
            for f in files:
                out_size += h5py.File(f, 'r')['data'].shape[0]

            # Assemble virtual dataset
            point_layout = h5py.VirtualLayout(shape=(out_size, 2048, 3))
            label_layout = h5py.VirtualLayout(shape=(out_size, 1))

            for i, f in enumerate(files):
                size = len(h5py.File(f, 'r')['data'])
                start_idx = i*2048
                end_idx = start_idx + size
                vsource_points = h5py.VirtualSource(f, 'data', shape=(size, 2048, 3), maxshape=(out_size, 2048, 3))
                vsource_label = h5py.VirtualSource(f, 'label', shape=(size, 1), maxshape=(out_size, 1))
                point_layout[start_idx:end_idx, ...] = vsource_points
                label_layout[start_idx:end_idx, ...] = vsource_label

            # Add virtual dataset to output file
            with h5py.File("data/{}_VDS.h5".format(prefix), 'w', libver='latest') as f:
                f.create_virtual_dataset('data', point_layout)
                f.create_virtual_dataset('label', label_layout)

        create_vds(train_files, 'train')
        create_vds(test_files, 'test')

        print("[Provider] Created Virtual Dataset.")


