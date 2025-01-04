import torch
import torchvision
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from io import BytesIO
import sys
import pygame
import time
import os
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydub import AudioSegment
from detect_obj.py import detect_and_extract
from anlayze_scene import predict_place
from pydub.playback import play



load_dotenv()
def play_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)


def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def image_to_audio(img):
    MODEL = os.environ.get("MODEL")
    client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))
       
    objs = detect_and_extract(img)
    place = predict_place(img)
    prompt = (os.environ.get("PROMPT").format(obj= objs, scene = place)

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
      
def main():

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
        cap.release()
        cv2.destroyAllWindows()

main()
