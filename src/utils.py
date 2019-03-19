# -*- coding: utf-8 -*-


import os
import pathlib
import sys

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import misc

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from preprocess import align


def load_image(image_path, width=None, height=None, mode='RGB'):
    image = Image.open(str(image_path))
    if width == None:
        width = image.width
    if height == None:
        height = image.height
    image = image.resize([width, height], Image.BILINEAR)
    image = np.array(image.convert(mode))
    return image


def reduce_dim(vectors, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    reduced = pca.fit_transform(vectors)
    # print(pca.explained_variance_ratio_)
    return reduced


def cluster_kmeans(vectors, labels, n_clusters,):
    kmeans = KMeans(n_clusters=n_clusters).fit(vectors)
    pred_labels = kmeans.predict(vectors)

    x = vectors[:, 0]
    y = vectors[:, 1]
    plt.scatter(x, y, c=labels)
    plt.colorbar()
    plt.show()

    return pred_labels
