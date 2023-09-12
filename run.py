import cv2
from detections.yolov5.detection import detection as detection1
from detections.customSSD.detection import detection as detection2

if __name__ == "__main__":

    img = cv2.imread("images/sample.png")
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res1 = detection1(img)
    res2 = detection2(img)
