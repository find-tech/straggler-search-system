import sys
import pickle
import cv2
import pathlib
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('tkagg')
from PIL import Image, ImageTk
import utils

min_score = 0.98

class ResultViewer(object):
    """Result Viewer:

    Args:
        result list
    """
    def __init__(self, results):
        self.results = results
        
    def print(self):
        print(self.results)

    def load_result(self, path='../result.pkl'):
        with open(path, 'rb') as f:
            self.results = pickle.load(f)

    def save_result(self, path='../result.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
        
    def save_txt(self, path='../result.txt'):
        with open(path, 'w') as f:
            f.writelines(str(self.results))

    def show_web(self):
        print('web:')
        print(self.results)

    def show_gui(self):
        root = tk.Tk()
        root.title("迷子検索システム  miccal")
        root.geometry("1280x1040")
        root.tk.call('tk', 'scaling', 1.5)
        gRoot.append(root)
        
        # Canvas Widget を生成
        canvas = tk.Canvas(root, width=2000)

        # Scrollbar を生成して配置
        bar = tk.Scrollbar(root, orient=tk.VERTICAL)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        bar.config(command=canvas.yview)

        # Canvas Widget を配置
        canvas.config(yscrollcommand=bar.set)
        canvas.config(scrollregion=(0,0,1920,1500)) #スクロール範囲
        canvas.pack(side=tk.LEFT, fill=tk.BOTH)

        frame = ResultFrame(master=canvas, results=self.results)

        # Frame Widgetを Canvas Widget上に配置
        canvas.create_window((0,0), window=frame, anchor=tk.NW, width=canvas.cget('width'))

        root.mainloop()

# 表示するイメージを保存する(表示されない対策)    
gImage = []

class ResultFrame(tk.Frame):
    def __init__(self, master, results):
        self.frame_back_color = '#333333'
        tk.Frame.__init__(self, master, width=1920, height=1500, bg=self.frame_back_color)
        self.cameras = results[0]
        self.data = results[1]
        self.createWidgets()
        self.master = master

    def createWidgets(self):
        outerFrame = tk.Frame(self, bg=self.frame_back_color)
        
        # ヘッダ
        headerFrame = tk.Frame(outerFrame, bg=self.frame_back_color)
        tk.Label(headerFrame, text='miccal  -  迷子検索結果', font=("", 25, "bold"), bg=self.frame_back_color, fg="white").pack(side=tk.LEFT, anchor=tk.W) # システム名
        tk.Button(headerFrame, text='地図表示', cursor="hand2", command=self.demo).pack(side=tk.RIGHT, anchor=tk.E, padx=7)
        headerFrame.pack(side=tk.TOP, fill=tk.BOTH)

        for datum in self.data:
            lineFrame = tk.Frame(outerFrame, padx=5, pady=5, bg=self.frame_back_color)
            lineFrame.place(relwidth=1.0)

            # 画像
            pilImage = Image.open("../" + datum['maigo']['image_path'])
            h, w = pilImage.size
            h2 = 160 if h < w else w * 160 // h
            w2 = 160 if w < h else h * 160 // w
            new_pilImage = pilImage.resize((w2, h2), resample=Image.BICUBIC)
            img =  ImageTk.PhotoImage(image=new_pilImage)
            gImage.append(img)
            canvas = tk.Canvas(lineFrame, width=160, height=160, bg="#ddd", highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=1)
            canvas.create_image(80, 80, image=img)

            # 迷子情報
            backColor = '#ddd'
            cameraFrame = tk.Frame(lineFrame, padx=5, pady=5, width=160, height=160, bg=backColor)
            cameraFrame.propagate(False)
            cameraFrame.place()
            
            tk.Label(cameraFrame, text="名前: " + datum['maigo']['maigo_name'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text=" ", bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="登録: " + datum['maigo']['datetime'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="地域: " + datum['maigo']['area'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="性別: " + datum['maigo']['gender'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="年齢: " + datum['maigo']['age'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            
            cameraFrame.pack(side=tk.LEFT, padx=1)
            
            # スペーサー
            spacer = tk.Label(lineFrame, text="", bg=self.frame_back_color)
            spacer.pack(side=tk.LEFT, padx=5)
            
            for people in datum['found_people'][:9]:
                score = people['score']
                backColor = 'red' if score < min_score else '#aaa'
                faceFrame = tk.Frame(lineFrame, relief=tk.SOLID, bd=0, padx=5, pady=2, width=96, height=160, bg=backColor)
                faceFrame.propagate(False)
                faceFrame.place()

                tag = people['camera_id'] + " " + str(people['index'])
                tk.Label(faceFrame, text=tag, bg=backColor).pack(side=tk.TOP) # カメラ名 + Index

                # 画像
                pilImage = Image.fromarray(people['image'])
                new_pilImage = pilImage.resize((80, 80), resample=Image.BICUBIC)
                img =  ImageTk.PhotoImage(image=new_pilImage)
                gImage.append(img)
                canvas = tk.Canvas(faceFrame, width=80, height=80, bg="black", cursor="hand2", highlightthickness=0)
                canvas.bind("<Button-1>", self.callback)
                canvas.widgetName = tag
                canvas.pack(side=tk.TOP)
                canvas.create_image(40, 40, image=img)

                tk.Label(faceFrame, text=people['datetime'].strftime('%m/%d %H:%M'), bg=backColor).pack(side=tk.TOP) # 日時
                tk.Label(faceFrame, text='{:.3f}'.format(score), bg=backColor).pack(side=tk.TOP) # スコア
                
                faceFrame.pack(side=tk.LEFT, padx=2)

            lineFrame.pack(anchor=tk.W)

        outerFrame.place(x=10, y=0)

    def demo(self):
        print("start demo")
        main_path = pathlib.Path().cwd().parent
        camera_path = '{}/map.png'.format(main_path)
        img = cv2.imread(camera_path)
        cv2.imshow("MAP", img)

    def callback(self, event):
        camera_name, face_id = event.widget.widgetName.split(" ")
        target = [c for c in self.cameras if camera_name == c.name][0]
        face = target.data.faces[int(face_id)]
        print("clicked :", target.pos, face['bounding_box'])
        main_path = pathlib.Path().cwd().parent
        camera_path = '{}/data/maigo_search/camera_data/dummy/{}_dummy.jpg'.format(main_path, target.name)
        img = cv2.imread(camera_path)
        img_height = img.shape[0]
        img_width = img.shape[1]

        x, y, w, h = face['bounding_box']
        m = w // 2 # 左右に2倍となるマージンにする
        # 切り取った画像を取得
        upper = y-m if y-m>0 else 0
        left = x-m if x-m>0 else 0
        bottom = img_height if img_height < y+h+m else y+h+m
        right = img_width if img_width < x+w+m else x+w+m

        cv2.rectangle(img, (left, upper), (right, bottom), (255, 0, 0), 2) # 青で描画
        drawed_path = '{}/data/maigo_search/camera_data/result.jpg'.format(main_path)
        cv2.imwrite(drawed_path,  img)
        cv2.destroyAllWindows()
        title = camera_name + "  -  Face" + face_id + "  -  " + target.data.date.strftime('%Y/%m/%d %H:%M')
        cv2.imshow(title, img)

        # データ作成して渡す
        time = str(target.data.date.isoformat())
        df = pd.DataFrame([[target.device, time]], columns=["device", "time"])
        utils.create_maigo_map(df)

# Windowインスタンス保存用 (代入しないと消えて落ちる)
gRoot = []

if __name__ == "__main__":
    rv = ResultViewer(None)
    rv.load_result()
    rv.show_gui()