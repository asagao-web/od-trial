import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import torch

from ssd_model import SSD
from profiler.profile import timefn
from utils.ssd_model import DataTransform

import os
# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# SSD300の設定
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSDネットワークモデル
model = SSD(phase="inference", cfg=ssd_cfg)

# SSDの学習済みの重みを設定
# net_weights = torch.load('./weights/ssd300_50.pth',
#                          map_location={'cuda:0': 'cpu'})

net_weights = torch.load(current_dir + '/weights/ssd300_mAP_77.43_v2.pth',
                        map_location={'cuda:0': 'cpu'})

model.load_state_dict(net_weights)
model.eval()
print('[Custom SSD] ネットワーク設定完了：学習済みの重みをロードしました')

# 3. 前処理クラスの作成
COLOR_MEAN = (104, 117, 123)  # (BGR)の色の平均値
INPUT_SIZE = 300  # 画像のinputサイズを300×300にする
transform = DataTransform(INPUT_SIZE, COLOR_MEAN)


@timefn
def detection(img:np.ndarray) -> np.ndarray:
    '''img is ndarray of shape (H, W, 3), BGR'''
    img_transformed, boxes, labels = transform(
    img, "val", "", "")  # アノテーションはないので、""にする
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = img.unsqueeze(0)
    return model(x)


if __name__ == "__main__":
    # img = np.zeros((640, 480, 3), dtype=np.uint8)
    # read from file
    img = cv2.imread("images/sample.png")
    # convert to rgb
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detection(img)
    print(">>>>>>>> RESULT SHAPE")
    print(res.shape)
    print(">>>>>>>> RESULT")
    print(res)



