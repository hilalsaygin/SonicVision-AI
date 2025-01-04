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
from playsound import playsound
import sys
import re
import multiprocessing
# import ray
from pydub import AudioSegment
from pydub.playback import play

import pygame
import time


load_dotenv()
def play_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)

def detect_and_extract(img):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', pretrained=True)
    # img = Image.open(image_path)
    # image = Image.open(BytesIO(image_path))

    # Perform inference
    results = model(img, size=640)
    # Get bounding box coordinates
    boxes = results.xyxy[0].cpu().numpy()
    class_names = model.names
    # Data structure to store information
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
        # cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.putText(img_cv2, f"{class_name}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    detected_objects.sort()
    # print(detected_objects)  
    # results.show()
    # cv2.waitKey(5000)  # Wait for 5 seconds (5000 milliseconds)
    # cv2.destroyAllWindows()
    return detected_objects

def predict_place(img):   ## png kabul etmiyor 
    
    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

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

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classe_names = list()
    with open(file_name) as class_file:
        for line in class_file:
            classe_names.append(line.strip().split(' ')[0][3:])
    classe_names = tuple(classe_names)
    
    
    # img = Image.open(image_path)
    # image = Image.open(BytesIO(image_path))
    
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    print(classe_names[idx[0]]) #return top prediction
    return (classe_names[idx[0]])
    

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def image_to_audio(img):
    # MODEL = "gpt-3.5-turbo-1106"
    MODEL = "gpt-4o"

    client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))
   
    # client = OpenAI(api_key= os.environ.get("test_key"))

    # result_yolo = multiprocessing.Queue()
    # result_cnn = multiprocessing.Queue()
    # yolo_process = multiprocessing.Process(target=detect_and_extract, args=(img, result_yolo)).start()

    # cnn_process = multiprocessing.Process(target=predict_place, args=(img, result_cnn)).start()
    # # Wait for the processes to finish
    # yolo_process.join()
    # cnn_process.join()
    # # Retrieve results from the queues
    # place = result_yolo.get()
    # objs = result_cnn.get()


    # place = predict_place(img)
    # objs = detect_and_extract(img)
    
    # objs, place = ray.get([detect_and_extract.remote(), predict_place.remote()])

    objs = detect_and_extract(img)
    place = predict_place(img)
    # objs, place = ray.get([ret_id1, ret_id2])
    prompt = ("consider a person took picture. using human-friendly language, describe the picture. desribe according to following information about the picture. there are objects in the image. here is the list of objects along with their coordinate information : {obj}."
    +"it is known that the image was taken in {scene}. generate description using the scene info and objects exist in the picture. include critical information about scene and positions but keep it brief. to describe object's positional relations, calculate the relational positions of objects using the coordinates but do not include numeric coordinate values directly. compare coordinates and define the location of the objects present in the image. focus on explaining relational positions of objects by stating which objects close to each other, next to each other ect. explain the image for a blind person using these information. do not use technical words. write a well defining description sentences.").format(obj= objs, scene = place)

    response = client.chat.completions.create(
    model=MODEL,
    messages=[
      {"role": "system", "content": "You are an assistant describing an image content for a blind person. use max 6 sentences."},
      {"role": "user", "content": prompt}
      
    ],
    )
    img_id = "cam_out"
    response_text= (response.choices[0].message.content)

    try:
        img_path = img.filename.split('.')[0].split('/')[-1]
        img_id = img_path
    except AttributeError:
        img_id = "exp_out"
    
    # Append a timestamp to ensure the filename is unique
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{img_id}_{timestamp}.mp3"

    speech_file_path = Path(__file__).parent / (unique_filename)
    res = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=response_text
    )
    

    res.stream_to_file(speech_file_path)
    play_audio(speech_file_path)
    # playsound(speech_file_path)

    # print(detect_and_extract(img_path))
    # print(response['choices'][0]['message']['content'])

      
def main():

    # img_path = "./images/10815824_2997e03d76.jpg"
    # if not os.access(img_path, os.W_OK):
    #     img_url = 'http://places.csail.mit.edu/demo/' + img_path.split('/')[-1]
    #     os.system('wget ' + img_url)

    # Initialize parser
    n = len(sys.argv)
    print("For input from library type in command line: -up <image path>")

    if (n > 1):

        upload = sys.argv[1]
        if (upload == '-up'):
            img_path = sys.argv[2]
            
            if not os.access(img_path, os.W_OK):
                img_url = 'http://places.csail.mit.edu/demo/' + img_path.split('/')[-1]
                os.system('wget ' + img_url)
            img = Image.open(img_path)
            img.show()
            # print(type(img))

            image_to_audio(img)
            
    else: #if not specied take input from webcam

        cap = cv2.VideoCapture(0)  # 0 indicates the default camera 
        counter = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            # Capture frame from webcam
            if counter % 59 == 0:
                counter = 1
            
                # Convert frame to PIL Image
                # img = Image.open(img_pil)
                # img_pil.show()
                # cv2.imshow('Current', frame)

                # detect_objects(img_path)
                # print(detect_and_extract(frame))
                # predict_place(frame)
                
                 # Resize the frame (e.g., to half of its original size)
                height, width = frame.shape[:2]
                new_width = int(width *1.3) 
                new_height = int(height*1.3)
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Display the resized frame
                cv2.imshow("Picture", resized_frame)
                cv2.waitKey(5000)
                img_pil = Image.fromarray(frame)
    
                image_to_audio(img_pil)
                cv2.waitKey(8000)

                cv2.destroyAllWindows()

                # Break the loop
                # when 'q' key is pressed
                if cv2.waitKey(3000) & 0xFF == ord('q'):
                    break
            else: 
                
                counter+=1
        # Release the webcam and close the OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

main()
