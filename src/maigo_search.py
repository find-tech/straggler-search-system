import csv
import datetime
import os
import pathlib
import sys

from PIL import Image
import cv2
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import misc

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

sss_path = os.path.abspath(os.path.join(os.path.curdir, os.pardir)) # straggler-search-system
sys.path.append(sss_path + '/src')

from facenet.src import facenet
from facenet.src.align import detect_face

from models import FaceNetModel
from preprocess import align
from camera import Camera
from database import MaigoDataBase
from show_result import ResultViewer


main_path = pathlib.Path().cwd().parent
model_path = main_path / 'models' / '20180402-114759' / '20180402-114759.pb'
maigo_db_path = main_path / 'configs' / 'maigo_db.csv'
camera_configs_path = main_path / 'configs' / 'camera_configs.csv'
img_extention = 'jpg'

class MaigoSearchEngine(object):
    """Search engine for Maingos.

    Args:
        model_path (str or pathlib.Path): Path to pretrained model.
        camera_condigs_path (str or pathlib.Path): Path to list of configs for cameras.
        threshold (float): Threshold to determine 2 images are similar or not. If the distance is less than this threshold, the images are thought to be similar.

    Attributes:
        model (FaceNetModel): FaceNet model.
        db_lostones (FaceImageDataBase): Database for lost people.
        cameras (list[Camera]): Cameras.
        threshold (float): Threshold to determine 2 images are similar or not. If the distance is less than this threshold, the images are thought to be similar.
    """
    def __init__(self, model_path, threshold=1.1):
        self.model = FaceNetModel(str(model_path))
        self.maigo_db = MaigoDataBase()
        self.cameras = []
        self.threshold = threshold

    def build_maigo_db(self, db_path):

        self.maigo_db.load(maigo_db_path)
        for maigo in self.maigo_db.people:
            print(maigo['maigo_name'])
            image, _ = align([main_path / maigo['image_path']])
            if len(image) == 0:
                raise ValueError("Image Not Found or Invalid Image: {}".format(maigo['image_path']))
            feature = self.model(image[0])
            maigo['feature']= feature
    
    def build_cameras(self, camera_configs_path):
        """Build cameras from config file.

        Args:
            camera_condigs_path (str or pathlib.Path): Path to list of configs for cameras.
        """
        configs = pd.read_csv(str(camera_configs_path))
        for i in range(len(configs)):
            config = configs.loc[i, :]
            if config.deleted == 1: # 削除済み
                continue
            name = str(config.camera_name)
            device = int(config.device)
            latitude = config.latitude
            longitude = config.longitude
            pos = (latitude, longitude)
            storage_path = main_path / config.storage_path
            camera = Camera(name, device, pos, storage_path, '../models/haarcascade_frontalface_default.xml', 'jpg')
            self.cameras.append(camera)
        return

    def search(self, query_vec, vectors, n=10,):
        """Search most similar vector from vectors to query_vec.

        Args:
            query_vec (numpy.ndarray): Query vector.
            vectors (numpy.ndarray): Searched vectors.
            n (int): Number of retrieved vectors.

        Returns:
            cands (list[dict]): Found ones.
        """
        scores = euclidean_distances(query_vec[np.newaxis, :], vectors)[0]
        indices = np.argsort(scores)[:n]
        scores = scores[indices]
        cands = []
        for idx in range(len(scores)):
            score = scores[idx]
            if score < self.threshold:
                cand = {
                    'score': score,
                    'index': indices[idx],
                    }
                cands.append(cand)
            else:
                break
        return cands

    def run(self):

        for camera in self.cameras:
            camera.shoot_dummy(str(main_path))
            #camera.start()
            #camera.shoot()
            #camera.stop()
            camera.data.save()  # if save, images are removed.
            features = []
            del_indices = []
            for i, face in enumerate(camera.data.faces):
                print(camera.name + "/face_" + str(i + 1))
                image, _ = align([str(face['path'])])
                if len(image) == 0:
                    del_indices.append(i)
                    continue
                image = image[0]
                features.append(self.model(image))
            for idx in del_indices[::-1]:
                del camera.data.faces[idx]
            camera.data.features = np.array(features)
        
        results_data = []
        for maigo in self.maigo_db.people:
            found_all = []            
            for camera in self.cameras:
                found_ones = self.search(maigo['feature'], camera.data.features, n=11,)
                if found_ones:
                    for person in found_ones:
                        person.update(camera.data.faces[person['index']])
                        person['image'] = cv2.cvtColor(cv2.imread(str(person['path'])), cv2.COLOR_BGR2RGB)
                        person['camera_id'] = camera.name
                        person['datetime'] = camera.data.date
                    
                    found_all.extend(found_ones)
                    
            found_all_sorted = sorted(found_all, key=lambda x:x['score'])
            result_data = {
                'maigo': maigo,
                'found_people': found_all_sorted,
                'shot_image': cv2.cvtColor(cv2.imread(str(camera.data.image_path)), cv2.COLOR_BGR2RGB),
                }
            results_data.append(result_data)
        results = [self.cameras, results_data]
        return results

if __name__ == "__main__":
    engine = MaigoSearchEngine(model_path, threshold=99)
    engine.build_maigo_db(maigo_db_path)
    engine.build_cameras(camera_configs_path)
    results = engine.run()
    rv =ResultViewer(results)
    rv.save_result()
    rv.show_gui()