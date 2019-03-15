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

from facenet.src import facenet
from facenet.src.align import detect_face


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


def align(image_paths, image_size=160, margin=44, gpu_memory_fraction=1.0):
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709 # scale factor

    # print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    aligned_indices = []
    aligned_images = []
    #aligned_images = [None] * len(image_paths)
    # aligned_image_paths = []
    for i,image_path in enumerate(image_paths):
        # print('%1d: %s' % (i, image_path))
        img = misc.imread(os.path.expanduser(str(image_path)))
        img_size = np.asarray(img.shape)[0:2]
        try:
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            # prewhitened = facenet.prewhiten(aligned)  # do in the FaceNetModel
            aligned_indices.append(i)
            aligned_images.append(aligned)
            #img_list[i] = prewhitened
            # aligned_image_paths.append(image_path)
        except:
            print('Cannot align: {}'.format(image_path))

    aligned_images = np.stack(aligned_images)
    return aligned_images, aligned_indices


class FaceNetModel(object):
    """FacenetのTensorFlow実装のwrapper。

    References:
        * [davidsandberg/facenet](https://github.com/davidsandberg/facenet)

    Args:
        model_path (str or pathlib.Path): モデルファイル（.pb）へのパス。

    Attributes:
        ...
    """
    def __init__(self, model_path,):
        facenet.load_model(str(model_path))
        self.input_image_size = 160
        self.sess = tf.Session()
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        self.embedding_size = self.embeddings.get_shape()[1]

    def __del__(self):
        self.sess.close()

    @staticmethod
    def load_image(image_path, width=None, height=None, mode='RGB'):
        image = Image.open(str(image_path))
        if width == None:
            width = image.width
        if height == None:
            height = image.height
        image = image.resize([width, height], Image.BILINEAR)
        image = np.array(image.convert(mode))
        return image

    def vectorize(self, image,):
        prewhitened = facenet.prewhiten(image)
        prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
        feed_dict = {
            self.phase_train_placeholder: False,
            self.images_placeholder: prewhitened,
            }
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        embeddings = embeddings.flatten()
        return embeddings

    
class ImageStorage(object):
    """

    """
    def __init__(self, model,):
        self.model = model
        self.image_paths = []
        self.labels = []
        self.labels_str = []
        self.images = []
        self.features = []
        self.size = 0
        self.label2num = {}

    def add(self, image_path, label):
        if label not in self.label2num:
            self.label2num[label] = len(self.label2num)
        aligned_images, aligned_indices = align(image_paths)
        if aligned_indices:
            self.image_paths.append(str(image_path))
            self.labels_str.append(label)
            self.labels.append(self.label2num[label])
            image = aligned_images[0]
            self.images.append(image)
            feature = self.model.vectorize(image)
            self.features.append(feature)
            self.size += 1

    def add_all(self, image_paths, labels):
        labels_id = []
        for label in labels:
            if label not in self.label2num:
                self.label2num[label] = len(self.label2num)
            labels_id.append(self.label2num[label])
        aligned_images, aligned_indices = align(image_paths)
        aligned_indices = np.array(aligned_indices)
        self.image_paths += np.array(image_paths)[aligned_indices].tolist()
        self.labels_str += np.array(labels)[aligned_indices].tolist()
        self.labels += np.array(labels_id)[aligned_indices].tolist()
        self.images += list(aligned_images)
        for image in aligned_images:
            feature = self.model.vectorize(image)
            self.features.append(feature)
        self.size = len(self.image_paths)

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

        # img_1 = FaceNetModel.load_image(path_1)
        # img_2 = FaceNetModel.load_image(path_2)
        img_1 = self.images[idx_1]
        img_2 = self.images[idx_2]

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

