# -*- coding: utf-8 -*-


import csv
import os
import pathlib
import sys

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import misc

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from preprocess import align


class MaigoDataBase(object):
    """Data strage for people with face image.

    Attributes:
        people (list[dict]): List of people data.
            People data has face image, name, ...
        feature_list (list): List of feature vectors.
            After build, this list will be empty
        features (numpy.ndarray): Feature vectors of the storaged people.
    """
    def __init__(self):
        self.people = []

    def add(self, person):
        self.people.append(person)

    def load(self, path, encoding='utf-8'):
        with open(str(path), 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.people.append(row)
        return


class ImageStorage(object):
    """

    """
    def __init__(self, model,):
        self.model = model
        self.image_paths = []
        self.labels = []
        self.labels_str = []
        # self.images = []
        self.features = []
        self.size = 0
        self.label2num = {}

    def add(self, image_path, label,):
        if label not in self.label2num:
            self.label2num[label] = len(self.label2num)
        images, extracted_filepaths = align([image_path], image_size=self.model.input_image_size, margin=44, gpu_memory_fraction=1.0)
        if not extracted_filepaths:
            return

        self.image_paths.append(str(image_path))
        self.labels_str.append(label)
        self.labels.append(self.label2num[label])
        # image = load_image(str(image_path), self.model.input_image_size, self.model.input_image_size, 'RGB')
        # self.images.append(image)
        feature = self.model(images[0])
        self.features.append(feature)
        self.size += 1

    def save(self, path, only_feature=True,):
        if only_feature:
            np.save(path, self.features)
        else:
            # パスとラベルとfeatureをまとめてcsvか何かに保存
            pass

    def compare(self, idx_1, idx_2, plot=True,):
        path_1 = str(self.image_paths[idx_1])
        path_2 = str(self.image_paths[idx_2])

        print('Path 1: {}'.format(path_1))
        print('Path 2: {}'.format(path_2))

        img_1 = load_image(path_1)
        img_2 = load_image(path_2)

        print('Shape 1: ', img_1.shape)
        print('Shape 2: ', img_2.shape)

        feature_1 = self.features[idx_1]
        feature_2 = self.features[idx_2]

        dist_euclid = euclidean_distances(feature_1, feature_2)[0, 0]
        print('Euclidian Distance: {}'.format(dist_euclid))

        cos_sim = cosine_similarity(feature_1, feature_2)[0, 0]
        print('Cosine Similarity: {}'.format(cos_sim))

        if plot:
            fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
            axL.imshow(img_1)
            axL.set_title('img_1')
            axR.imshow(img_2)
            axR.set_title('img_2')
            plt.show()

        return

    def most_similar(self, idx_1, idx_2, metrics='cosine'):
        pass
