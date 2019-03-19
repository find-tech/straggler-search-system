# -*- coding: utf-8 -*-


import csv
import datetime
import os
import pathlib
import sys

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import align



class CameraData(object):
    """Data about cameras:

    Args:
        storage_path (str or pathlib.Path): Pathv to storage images temporarily.
        extention (str): Extention of saved images (without dot).
            Defaults to 'jpg'.

    Attributes:
        storage_path (str or pathlib.Path): Pathv to storage images temporarily.
        extention (str): Extention of saved images (without dot).
        image_path (str or pathlib.Path): Path to save the shot image.
        image (numpy.ndarray): Shot image.
        date (datetime.datetime): Date of the shot.
        face_data (list[dict]): List of face data.
            Face data has the cropped image and the coordinates of bounding box.
            face = {
                'image': numpy.ndarray,
                'bounding_box': cv2.rect,
                'path': str or pathlib.Path,
                }
        features (np.ndarray): Feature vectors of the faces.
    """
    def __init__(self, storage_path, extention='jpg'):
        self.storage_path = pathlib.Path(storage_path)
        self.extention = extention
        self.image_path = self.storage_path / 'image.{}'.format(self.extention)
        self.image = None
        self.date = None
        self.faces = []
        self.features = None

    def __len__(self):
        return len(self.faces)

    def save(self):
        """Save shot image and cropped face images."""
        cv2.imwrite(str(self.image_path), self.image)
        self.image = None

        n_face = len(self.faces)
        for i in range(n_face):
            img_path = self.storage_path / 'face_{}.{}'.format(i+1, self.extention)
            self.faces[i]['path'] = img_path
            cv2.imwrite(str(img_path), self.faces[i]['image'])
            self.faces[i]['image'] = None

    def reset(self):
        """Delete storaged and saved images."""
        for img_path in self.storage_path.glob('*.{}'.format(self.extention)):
            img_path.unlink()
        self.image = None
        self.date = None
        self.faces = []
        self.features = None


class Camera(object):
    """Camera.

    Args:
        name (str): Name of the camera.
        device (int): Device number of the camera.
        pos (tuple(float, float)): Position of the camera,
            represented as (latitude, longtitude)
        storage_path (str or pathlib.Path): Pathv to storage images temporarily.
        extention (str): Extention of saved images (without dot).
            Defaults to 'jpg'.

    Attributes:
        name (str): Name of the camera.
        device (int): Device number of the camera.
        pos (tuple(float, float)): Position of the camera,
            represented as (latitude, longtitude)
        data (CameraData): Data of the shot image.
    """
    def __init__(self, name, device, pos, storage_path, cascade_file, extention='jpg'):
        self.name = name
        self.device = device
        self.pos = pos
        self.cascade_file = cascade_file
        self.data = CameraData(storage_path, extention)

    def start(self):
        self.cap = cv2.VideoCapture(self.device)

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def shoot(self, mirror=True, size=None):
        """Capture video from camera

        """
        self.data.date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Webカメラから画像を1枚読み込む
        _, frame = self.cap.read()

        # 鏡のように映るか否か
        if mirror:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        self.data.image = frame
        frame = self.process(frame)

        # 画面を表示する
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(10) # 10 msec待つ
        if k == 27: # ESCキーで終了
            return False
        return True

    def shoot_dummy(self):
        self.data.date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.name == 'Camera1':
            path = '/Users/taro/Documents/straggler-search-system/data/maigo_search/camera_data/dummy/Camera1_dummy.jpg'
        else:
            path = '/Users/taro/Documents/straggler-search-system/data/maigo_search/camera_data/dummy/Camera2_dummy.jpg'
        frame = cv2.imread(path)
        self.data.image = frame
        frame = self.process(frame)
        return

    def process(self, frame, margin=80,):
        # グレースケールに変換 <<<---------------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔検出用カスケードファイルの読み込み
        face_cascade = cv2.CascadeClassifier(self.cascade_file)

        # カスケードファイルで複数の顔を検出
        face_bbs = face_cascade.detectMultiScale(gray)

        # 顔の周りの余白
        m = margin
        for face_bb in face_bbs:
            x, y, w, h = face_bb
            # 切り取った画像を取得
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            upper = y-m if y-m>0 else 0
            left = x-m if x-m>0 else 0
            cropped_image = rgb_image[upper:y+h+m, left:x+w+m, :]

            face = {
                'image': cropped_image,
                'bounding_box': face_bb,
                'path': None,
                }
            self.data.faces.append(face)

            #顔部分を四角で囲う
            frame = cv2.rectangle(frame,(x-m,y-m),(x+w+m,y+h+m),(255,0,0),2)

        return frame
