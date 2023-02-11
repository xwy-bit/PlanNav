# add parent dir to import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.preprocess import PreProcess
import cv2
import torch

pre_process = PreProcess('resnet50')

test_img = cv2.imread('test/asset/test_img00.png')
test_img = torch.Tensor(test_img).permute(2, 0, 1) # To [C , H , W]

print(test_img.shape)

result = pre_process.go(test_img)

print('[SHAPE] : ',result.shape)