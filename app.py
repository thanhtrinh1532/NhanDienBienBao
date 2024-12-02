from flask import Flask, render_template, Response
import numpy as np
import cv2
from tensorflow import keras
import os

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn đến mô hình
model_path = 'D:/MLearn/gtsrb-main/traffic_sign_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Tải mô hình đã lưu
model = keras.models.load_model(model_path)

threshold = 0.75  # THRESHOLD


# Định nghĩa hàm tiền xử lý hình ảnh
def preprocess_img(imgBGR, erode_dilate=True):
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

    # Thiết lập ngưỡng màu
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin


def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def getClassName(classNo):
    classes = [
        'Gioi han toc do 20 km/h', 'Gioi han toc do 30 km/h', 'Gioi han toc do 50 km/h',
        'Gioi han toc do 60 km/h', 'Gioi han toc do 70 km/h', 'Gioi han toc do 80 km/h',
        'Het gioi han toc do 80 km/h', 'Gioi han toc do 100 km/h', 'Gioi han toc do 120 km/h',
        'Cam vuot', 'Cam vuot xe tren 3.5 tan', 'Quyen uu tien tai giao lo tiep theo',
        'Duong uu tien', 'Nhuong duong', 'Dung lai', 'Cam tat ca cac loai xe',
        'Cam xe tren 3.5 tan', 'Cam vao', 'Canh bao chung', 'Khuc cua nguy hiem ben trai',
        'Khuc cua nguy hiem ben phai', 'Duong cong kep', 'Duong go ghe', 'Duong tron',
        'Duong hep ben phai', 'Cong truong', 'Den tin hieu', 'Nguoi di bo',
        'Tre em qua duong', 'Xe dap qua duong', 'Canh bao bang/ tuyet',
        'Dong vat hoang da qua duong', 'Het moi gioi han toc do va vuot',
        'Re phai phia truoc', 'Re trai phia truoc', 'Chi di thang',
        'Di thang hoac re phai', 'Di thang hoac re trai', 'Di ve ben phai',
        'Di ve ben trai', 'Bat buoc di vong', 'Het cam vuot', 'Het cam vuot xe tren 3.5 tan'
    ]
    return classes[classNo] if classNo < len(classes) else "Unknown"


# Định nghĩa hàm tạo video
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_bin = preprocess_img(img, False)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)

        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(img.shape[1], int(xc + size / 2))
            y2 = min(img.shape[0], int(yc + size / 2))

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop_img = preprocessing(crop_img)
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = crop_img.reshape(1, 32, 32, 1)
                predictions = model.predict(crop_img)
                classIndex = np.argmax(predictions[0])
                probabilityValue = predictions[0][classIndex]
                if probabilityValue > threshold:
                    className = getClassName(classIndex)
                    label = f"{className} ({probabilityValue:.2f})"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
