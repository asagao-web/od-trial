import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from profiler.profile import timefn


model = ssd300_vgg16(pretrained = True)
model.eval()


@timefn
def detection(img:np.ndarray) -> np.ndarray:
    #convert input image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    return model(img_tensor)

if __name__ == "__main__":
    import cv2
    img = cv2.imread("images/sample.png")
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detection(img)
    print(dir(res))
    print(res)