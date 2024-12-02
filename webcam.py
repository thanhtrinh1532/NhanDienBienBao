import numpy as np
import cv2
from tensorflow import keras

threshold = 0.75  # THRESHOLD
font = 'D:/MLearn/gtsrb-main/Arial.ttf'

model = keras.models.load_model('traffic_sign_model.h5')


def preprocess_img(imgBGR, erode_dilate=True):  # pre-processing for detecting signs in image.
    rows, cols, _ = imgBGR.shape

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
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


def draw_rects_on_img(img, rects):
    img_copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):  # Corrected function name
    if classNo == 0:
        return 'Gioi han toc do 20 km/h'
    elif classNo == 1:
        return 'Gioi han toc do 30 km/h'
    elif classNo == 2:
        return 'Gioi han toc do 50 km/h'
    elif classNo == 3:
        return 'Gioi han toc do 60 km/h'
    elif classNo == 4:
        return 'Gioi han toc do 70 km/h'
    elif classNo == 5:
        return 'Gioi han toc do 80 km/h'
    elif classNo == 6:
        return 'Het gioi han toc do 80 km/h'
    elif classNo == 7:
        return 'Gioi han toc do 100 km/h'
    elif classNo == 8:
        return 'Gioi han toc do 120 km/h'
    elif classNo == 9:
        return 'Cam vuot'
    elif classNo == 10:
        return 'Cam vuot xe tren 3.5 tan'
    elif classNo == 11:
        return 'Quyen uu tien tai giao lo tiep theo'
    elif classNo == 12:
        return 'Duong uu tien'
    elif classNo == 13:
        return 'Nhuong duong'
    elif classNo == 14:
        return 'Dung lai'
    elif classNo == 15:
        return 'Cam tat ca cac loai xe'
    elif classNo == 16:
        return 'Cam xe tren 3.5 tan'
    elif classNo == 17:
        return 'Cam vao'
    elif classNo == 18:
        return 'Canh bao chung'
    elif classNo == 19:
        return 'Khuc cua nguy hiem ben trai'
    elif classNo == 20:
        return 'Khuc cua nguy hiem ben phai'
    elif classNo == 21:
        return 'Duong cong kep'
    elif classNo == 22:
        return 'Duong go ghe'
    elif classNo == 23:
        return 'Duong tron'
    elif classNo == 24:
        return 'Duong hep ben phai'
    elif classNo == 25:
        return 'Cong truong'
    elif classNo == 26:
        return 'Den tin hieu'
    elif classNo == 27:
        return 'Nguoi di bo'
    elif classNo == 28:
        return 'Tre em qua duong'
    elif classNo == 29:
        return 'Xe dap qua duong'
    elif classNo == 30:
        return 'Canh bao bang/ tuyet'
    elif classNo == 31:
        return 'Dong vat hoang da qua duong'
    elif classNo == 32:
        return 'Het moi gioi han toc do va vuot'
    elif classNo == 33:
        return 'Re phai phia truoc'
    elif classNo == 34:
        return 'Re trai phia truoc'
    elif classNo == 35:
        return 'Chi di thang'
    elif classNo == 36:
        return 'Di thang hoac re phai'
    elif classNo == 37:
        return 'Di thang hoac re trai'
    elif classNo == 38:
        return 'Di ve ben phai'
    elif classNo == 39:
        return 'Di ve ben trai'
    elif classNo == 40:
        return 'Bat buoc di vong'
    elif classNo == 41:
        return 'Het cam vuot'
    elif classNo == 42:
        return 'Het cam vuot xe tren 3.5 tan'


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # Check if the video capture opened successfully
        print("Error: Could not open video.")
        exit()

    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, img = cap.read()
        if not ret:  # Check if frame is read correctly
            print("Error: Could not read frame.")
            break

        img_bin = preprocess_img(img, False)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)  # get x,y,h and w
        img_bbx = img.copy()

        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size > 0:  # Ensure crop_img is valid
                crop_img = preprocessing(crop_img)
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = crop_img.reshape(1, 32, 32, 1)  # Add the batch dimension
                predictions = model.predict(crop_img)
                classIndex = np.argmax(predictions[0])
                probabilityValue = predictions[0][classIndex]
                if probabilityValue > threshold:  # Only display if probability exceeds threshold
                    className = getClassName(classIndex)
                    label = f"{className} ({probabilityValue:.2f})"
                    cv2.rectangle(img_bbx, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_bbx, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

        cv2.imshow("Traffic Sign Detection", img_bbx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
