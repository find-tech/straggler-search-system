# -*- coding: utf-8 -*-


import os
import pathlib
import sys

from PIL import Image
import numpy as np
import tensorflow as tf

from scipy import misc

from facenet.src import facenet
from facenet.src.align import detect_face


def align(image_paths, image_size=160, margin=32, gpu_memory_fraction=1.0):
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
        try:
            img = misc.imread(str(image_path))
            img = img[:,:,0:3] # apply for 32bit image
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            if len(bounding_boxes) == 0:
                print('No bounding boxes: {}'.format(image_path))
                continue
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
    if 0 < len(aligned_images):
        aligned_images = np.stack(aligned_images)
    return aligned_images, aligned_indices
