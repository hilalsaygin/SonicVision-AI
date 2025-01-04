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
from dotenv import load_dotenv
from pathlib import Path
import sys
import multiprocessing
from pydub import AudioSegment
from pydub.playback import play

import pygame
import time

load_dotenv()


def predict_place(img):   # pass jpeg file for frames from video input 
    
    arch = 'resnet18'

    # load the moel and labels
    model_file = '%s_places365.pth.tar' % arch
    file_name = 'categories_places365.txt'


    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classe_names = list()
    with open(file_name) as class_file:
        for line in class_file:
            classe_names.append(line.strip().split(' ')[0][3:])
    classe_names = tuple(classe_names)
      
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    print(classe_names[idx[0]]) #return top prediction
    return (classe_names[idx[0]])
    
