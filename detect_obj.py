import torch
import torchvision
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from io import BytesIO
import os
import cv2
import numpy as np
import time
from openai import OpenAI
from pathlib import Path
import sys
import time


def detect_and_extract(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', pretrained=True)

    # Perform inference
    results = model(img, size=640)
    boxes = results.xyxy[0].cpu().numpy()
    class_names = model.names
    detected_objects = []
    # img_cv2 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Compare relative positions
    for i in range(len(boxes)):
        box_info = []
        box1 = boxes[i]
        # Access class names using class labels
        c1 = class_names[int(box1[-1])]

        # Add class name and coordinates to the data structure
        cords = (round(box1[2],1),round(box1[3],1))

        box_info.append(cords)
        detected_objects.append(box_info)
        box_info.append(c1)


    detected_objects.sort()

    return detected_objects

