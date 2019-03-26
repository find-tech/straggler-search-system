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
            bgr_frame = cv2.cvtColor(self.faces[i]['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), bgr_frame)
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
        del self.cap
        self.cap = None
        
    def shoot(self, mirror=True, size=None):
        """Capture video from camera

        """
        self.data.date = datetime.datetime.now()

        # Webカメラから画像を1枚読み込む
        _, frame = self.cap.read()

        # 鏡のように映るか否か
        if mirror:
            frame = frame[:,::-1]

        # フレームをリサイズ アスペクト比を考慮
        if frame.shape == (480, 640, 3):
            size = (1024, 768)
        else:
            size = (1024, 576)
        frame = cv2.resize(frame, size)

        self.data.image = frame
        frame = self.process(frame)

        # 画面を表示する
        cv2.imshow('camera capture', frame)

        _ = cv2.waitKey(1000) # 1000 msec待つ
        
        hasFace = 0 < len(self.data.faces)
        return hasFace

    def shoot_dummy(self, main_path):
        self.data.date = datetime.datetime.now()

        path = '{}/data/maigo_search/camera_data/dummy/{}_dummy.jpg'.format(main_path, self.name)
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError("Can't Read Camera Image: {}".format(path))
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
        img_height = self.data.image.shape[0]
        img_width = self.data.image.shape[1]
        cropped = []
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for face_bb in face_bbs:
            x, y, w, h = face_bb
            m = w // 2 # 左右に2倍となるマージンにする
            # 切り取った画像を取得
            upper = y-m if y-m>0 else 0
            left = x-m if x-m>0 else 0
            bottom = img_height if img_height < y+h+m else y+h+m
            right = img_width if img_width < x+w+m else x+w+m
            cropped_image = rgb_image[upper:bottom, left:right, :]

            face = {
                'image': cropped_image,
                'bounding_box': face_bb,
                'path': None,
                }
            self.data.faces.append(face)
            cropped.append((left, upper, right, bottom))
        
        for face_cropped in cropped:
            #顔部分を四角で囲う
            (left, upper, right, bottom) = face_cropped
            cv2.rectangle(frame, (left, upper), (right, bottom), (255, 0, 0), 2) # 青で描画

        return frame

if __name__ == "__main__":

    # 秋葉原にあるカメラでキャプチャ実行(という設定)
    camera = Camera("Test Camera", 0, (35.7, 139.7), 'Camera_test', '../models/haarcascade_frontalface_default.xml')
    camera.start()
    for _ in range(50):
        # サイズ指定しないと落ちる
        camera.shoot()
    camera.stop()
