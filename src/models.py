# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from facenet.src import facenet


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

    def __call__(self, image):
        return self.vectorize(image)

    def vectorize(self, image):
        prewhitened = facenet.prewhiten(image)
        prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
        feed_dict = {
            self.images_placeholder: prewhitened,
            self.phase_train_placeholder: False,
            }
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        embeddings = embeddings.flatten()
        return embeddings
