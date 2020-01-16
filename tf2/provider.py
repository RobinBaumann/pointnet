import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import open3d as o3d
import os
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'ModelNet10')):
    www = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


class PointCloudProvider(Sequence):
    """
    Lazily load point clouds and annotations from filesystem and prepare it for model training.
    """

    def __init__(self, dataset, batch_size, n_classes, sample_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.sample_size = sample_size

        # indices of samples dictionary used for shuffling
        self.indices = np.arange(len(dataset))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_samples = self.dataset.iloc[batch_indices]

        return self.__generate_data(batch_samples)

    def __generate_data(self, batch_samples):
        X = []
        y = []
        for i, row in batch_samples.iterrows():
            try:
                mesh = o3d.io.read_triangle_mesh((row["path"]))
                pcd = mesh.sample_points_uniformly(number_of_points=self.sample_size)
                points = np.asarray(pcd.points)
                centered_points = (points - points.mean(axis=0))
                normalized_points = centered_points / centered_points.max()
                X.append(normalized_points)
                y.append(row["class"])
            except:
                continue

        return self.rotate_point_clouds(np.array(X)), to_categorical(np.array(y), num_classes=self.n_classes)

    def on_epoch_end(self):
        """Shuffle training data, so batches are in different order"""
        np.random.shuffle(self.indices)

    def rotate_point_clouds(self, batch, rotation_angle_range=(-np.pi / 8, np.pi / 8)):
        """Rotate point cloud around y-axis (=up) by random angle"""
        for b, pc in enumerate(batch):
            phi = np.random.uniform(*rotation_angle_range)
            c, s = np.cos(phi), np.sin(phi)
            R = np.matrix([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]])
            batch[b, :, :3] = np.dot(pc[:, :3], R)
        return batch

    @staticmethod
    def initialize_dataset():
        """
        Loads an index to all files and structures them.
        :param data_directory: directory containing the data files
        :param file_extension: extension of the data files
        :return: pandas dataframe containing an index to all files and a label index,
            mapping numerical label representations to label names.
        """

        data = os.path.join(DATA_DIR, "ModelNet10/")

        files = [
            os.path.join(r, f)
            for r, d, fs in os.walk(data)
            for f in fs if f.endswith('.off')
        ]

        dataframe = pd.DataFrame({
            "path": files,
            "class": pd.Categorical([f.rsplit("/", 3)[1] for f in files]),
            "is_train": ["train" in f for f in files]
        })

        factorization = dataframe["class"].factorize()
        dataframe["class"] = factorization[0]

        return dataframe, factorization[1]
