# conda install opencv -c conda-forge
# curl -o ../models/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

import cv2
def capture_camera(mirror=True, size=None):
    """Capture video from camera"""

    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    while True:
        
        # Webカメラから画像を1枚読み込む
        _, frame = cap.read()

        # 鏡のように映るか否か
        if mirror:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)
        
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔検出用カスケードファイルの読み込み
        face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

        # カスケードファイルで複数の顔を検出
        faces = face_cascade.detectMultiScale(gray)

        # 顔の周りの余白
        m = 80
        
        for (x,y,w,h) in faces:

            # 切り取った画像を取得
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_image = rgb_image[y-m:y+h+m,x-m:x+w+m,:]
            
            # ここでEmbeddingする
            # ...

            #顔部分を四角で囲う
            frame = cv2.rectangle(frame,(x-m,y-m),(x+w+m,y+h+m),(255,0,0),2)
        
        # 画面を表示する
        cv2.imshow('camera capture', frame)       

        k = cv2.waitKey(10) # 10 msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

# capture_camera()
capture_camera(True, (800,600)) # サイズを指定しないとエラーになる環境がある
