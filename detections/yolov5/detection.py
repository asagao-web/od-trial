import numpy as np
import torch
from profiler.profile import timefn

# Model, initial loading from internet
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# make script to upgrade model and save for next time


@timefn
def detection(img:np.ndarray) -> np.ndarray:
    '''img is ndarray of shape (H, W, 3), RGB'''
    return model(img)


if __name__ == "__main__":
    # img = np.zeros((640, 480, 3), dtype=np.uint8)
    # read from file
    import cv2
    img = cv2.imread("images/sample.png")
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detection(img)
    print(dir(res))
    print(res.xyxyn)