import sys
import pickle
import tkinter as tk
from PIL import Image, ImageTk

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
        root.title("Search Result")
        gRoot.append(root)
        f = ResultFrame(master=root, data=self.results)
        f.pack()
        f.mainloop()

# 表示するイメージを保存する(表示されない対策)    
gImage = []

class ResultFrame(tk.Frame):
    def __init__(self, master, data):
        tk.Frame.__init__(self, master, width=1540, height=1200)
        self.data = data
        self.createWidgets()
        self.master = master

    def createWidgets(self):
        outerFrame = tk.Frame(self)
        tk.Label(outerFrame, text='迷子検索結果').pack(side=tk.TOP) # システム名
        

        for datum in self.data:
            lineFrame = tk.Frame(outerFrame, padx=5, pady=5)
            lineFrame.place(relwidth=1.0)

            # カメラ
            cameraFrame = tk.Frame(lineFrame, relief=tk.RIDGE, bd=2, padx=5, pady=5, width=200, height=200, bg='darkgray')
            cameraFrame.propagate(False)
            cameraFrame.place()

            tk.Label(cameraFrame, text=datum['camera_id']).pack(side=tk.TOP) # カメラ名
            tk.Label(cameraFrame, text=datum['datetime']).pack(side=tk.TOP) # 日時
            tk.Label(cameraFrame, text=datum['maigo']['maigo_name']).pack(side=tk.TOP) # 名前
            tk.Label(cameraFrame, text=datum['maigo']['age']).pack(side=tk.TOP) # 年齢
            tk.Label(cameraFrame, text=datum['maigo']['area']).pack(side=tk.TOP) # 地域
            tk.Label(cameraFrame, text=datum['maigo']['gender']).pack(side=tk.TOP) # 性別
            tk.Label(cameraFrame, text=datum['maigo']['image_path']).pack(side=tk.TOP) # 画像
            
            cameraFrame.pack(side=tk.LEFT)

            for people in datum['found_people']:
                score = people['score']
                backColor = 'red' if score < 0.85 else 'darkgray'
                faceFrame = tk.Frame(lineFrame, relief=tk.RIDGE, bd=2, padx=5, pady=5, width=120, height=200,bg=backColor)
                faceFrame.propagate(False)
                faceFrame.place()

                tk.Label(faceFrame, text=str(people['bounding_box'])).pack(side=tk.TOP) # BBox
                tk.Label(faceFrame, text=str(people['index'])).pack(side=tk.TOP) # スコア
                tk.Label(faceFrame, text=str(people['path'])).pack(side=tk.TOP) # 画像パス
                tk.Label(faceFrame, text=str(score)).pack(side=tk.TOP) # スコア

                # 画像
                import numpy as np
                pilImage = Image.fromarray(people['image'])
                img =  ImageTk.PhotoImage(image=pilImage)
                gImage.append(img)
                canvas = tk.Canvas(faceFrame, width=100, height=100, bg="black")
                canvas.pack(side=tk.TOP)
                canvas.create_image(50, 50, image=img)
                
                faceFrame.pack(side=tk.LEFT)

            lineFrame.pack(anchor=tk.W)

        outerFrame.place(x=0, y=0)

# Windowインスタンス保存用 (代入しないと消えて落ちる)
gRoot = []

if __name__ == "__main__":
    rv = ResultViewer(None)
    rv.load_result()
    rv.show_gui()