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

import pandas as pd
import folium
import iso8601
import webbrowser

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

def create_maigo_map(df_maigo_position):
    # カメラコンフィグの読み取り
    df_camera_config = pd.read_csv("../configs/camera_configs.csv")

    # map表示用dfの作成
    df_maigo_position.columns = ["device", "time"]
    df_maigo_positon_time = pd.merge(df_maigo_position, df_camera_config, on = "device").loc[:, ["time", "latitude", "longitude"]]

    # mapの作成
    # mapの初期位置の定義
    _map = folium.Map(location=[df_maigo_positon_time.loc[0, "latitude"], df_maigo_positon_time.loc[0, "longitude"]], zoom_start=20)

    # 地図へマーカーを付与
    for _, row in df_maigo_positon_time.iterrows():
        time = iso8601.parse_date(row["time"]).strftime('%Y-%m-%d %H:%M:%S')
        folium.Marker([row["latitude"], row["longitude"]], popup = time).add_to(_map)

    # 作成したmapをviewに作成
    _map.save('../view/map.html')
    url = pathlib.Path().cwd() / '../view/map.html'
    webbrowser.get("chrome").open(str(url))

if __name__ == "__main__":
    df_camera_config = pd.read_csv("../configs/test_data_map.csv")
    create_maigo_map(df_camera_config)