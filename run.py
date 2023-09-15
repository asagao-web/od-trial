import cv2
from detections.yolov5.detection import detection as detection1
from detections.customSSD.detection import detection as detection2
from detections.fast_r_cnn.detection import detection as detection3
from detections.ssd300.detection import detection as detection4
from detections.ssd300lite.detection import detection as detection5
from detections.fast_r_cnn_with_mobilenet.detection import detection as detection6

"""
検証されたモデル
yolo v5
custom SSD
fast r cnn with resnet 50
normal SSD
SSD lite (SSD with mobilenet)
fast r cnn with mobilenet
"""

if __name__ == "__main__":

    img = cv2.imread("C:\Python\od-trial\images\sample.png")
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res1 = detection1(img)
    res2 = detection2(img)
    res3 = detection3(img)
    res4 = detection4(img)
    res5 = detection5(img)
    res6 = detection6(img)

