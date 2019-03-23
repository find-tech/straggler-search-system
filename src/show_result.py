import sys
import pickle
import tkinter as tk
import numpy as np
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
        root.geometry("1358x1040")
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
        canvas.config(scrollregion=(0,0,1600,1500)) #スクロール範囲
        canvas.pack(side=tk.LEFT, fill=tk.BOTH)

        frame = ResultFrame(master=canvas, data=self.results)

        # Frame Widgetを Canvas Widget上に配置
        canvas.create_window((0,0), window=frame, anchor=tk.NW, width=canvas.cget('width'))

        root.mainloop()

# 表示するイメージを保存する(表示されない対策)    
gImage = []

class ResultFrame(tk.Frame):
    def __init__(self, master, data):
        self.frame_back_color = '#333333'
        tk.Frame.__init__(self, master, width=1600, height=1500, bg=self.frame_back_color)
        self.data = data
        self.createWidgets()
        self.master = master

    def createWidgets(self):
        outerFrame = tk.Frame(self, bg=self.frame_back_color)
        tk.Label(outerFrame, text='迷子検索結果', font=("",25), bg=self.frame_back_color, fg="white").pack(side=tk.TOP, anchor=tk.NW) # システム名
        
        for datum in self.data:
            lineFrame = tk.Frame(outerFrame, padx=5, pady=5, bg=self.frame_back_color)
            lineFrame.place(relwidth=1.0)

            # 画像
            pilImage = Image.open("../" + datum['maigo']['image_path'])
            #h, w = pilImage.size
            new_pilImage = pilImage.resize((147, 147), resample=Image.BICUBIC)
            img =  ImageTk.PhotoImage(image=new_pilImage)
            gImage.append(img)
            canvas = tk.Canvas(lineFrame, width=144, height=144, bg="black")
            canvas.pack(side=tk.LEFT)
            canvas.create_image(72, 72, image=img)

            # 迷子情報
            backColor = '#ddd'
            cameraFrame = tk.Frame(lineFrame, padx=5, pady=5, width=144, height=144, bg=backColor)
            cameraFrame.propagate(False)
            cameraFrame.place()
            
            tk.Label(cameraFrame, text="名前: " + datum['maigo']['maigo_name'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="年齢: " + datum['maigo']['age'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="地域: " + datum['maigo']['area'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="性別: " + datum['maigo']['gender'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            tk.Label(cameraFrame, text="登録: " + datum['maigo']['datetime'], bg=backColor).pack(side=tk.TOP, anchor=tk.NW)
            
            cameraFrame.pack(side=tk.LEFT, padx=1)
            
            # スペーサー
            spacer = tk.Label(lineFrame, text="", bg=self.frame_back_color)
            spacer.pack(side=tk.LEFT, padx=5)
            
            for people in datum['found_people'][:10]:
                score = round(people['score'], 3)
                backColor = 'red' if score < 0.85 else '#aaa'
                faceFrame = tk.Frame(lineFrame, relief=tk.SOLID, bd=0, padx=5, pady=5, width=94, height=144, bg=backColor)
                faceFrame.propagate(False)
                faceFrame.place()

                # 画像
                pilImage = Image.fromarray(people['image'])
                new_pilImage = pilImage.resize((83, 83), resample=Image.BICUBIC)
                img =  ImageTk.PhotoImage(image=new_pilImage)
                gImage.append(img)
                canvas = tk.Canvas(faceFrame, width=80, height=80, bg="black")
                canvas.pack(side=tk.TOP)
                canvas.create_image(40, 40, image=img)

                tk.Label(faceFrame, text=people['camera_id'] + " #" + str(people['index']), bg=backColor).pack(side=tk.TOP) # カメラ名 + Index
                tk.Label(faceFrame, text=str(score), bg=backColor).pack(side=tk.TOP) # スコア
                tk.Label(faceFrame, text=people['datetime'], bg=backColor).pack(side=tk.TOP) # 日時
                
                faceFrame.pack(side=tk.LEFT, padx=4)

            lineFrame.pack(anchor=tk.W)

        outerFrame.place(x=0, y=0)

# Windowインスタンス保存用 (代入しないと消えて落ちる)
gRoot = []

if __name__ == "__main__":
    rv = ResultViewer(None)
    rv.load_result()
    rv.show_gui()