
# SonicVision-AI

## Overview
SonicVision-AI is an advanced AI-driven project designed for real-time visual recognition and processing. An Audio Description Generator to narrate Video contents using Deep Learning.
 The system is built to detect objects, and analyze the scene of the video input. It generates description pragraph to define currently captured frame including both relative position of objects to each other and the place the frame captured. The descripion is converted using text-to-speech AI modules. The surroundings described vocally.

## Features
- **Real-Time Video Processing**: The system captures and processes video frames for futher processing.
- **Visual Recognition**: Implements visual recognition using state-of-the-art deep learning models.
- **AI-Driven Interactions**: AI algorithms interact with coth images and live streaming video inputs for dynamic responses.
- **YOLOv5 model**: Precise object detection withn positional information.
- **Fine-tuning CNN deep leanring model for place prediction**: places356 dataset covering major scene categories. 

## Requirements
To run this project, ensure you have the following installed:
- Python 3.x
- Required Python libraries from `requirements.txt`
- Audio and camera hardware for input

## Installation

### Clone the Repository
```bash
git clone <repo-link>
cd SonicVision-AI
```

### Install Dependencies
Install the necessary Python libraries with pip:
```bash
chmod +x install.sh
./install.sh
```
#### Note that for scene analysis CNN-MODEL file path needs to be provided in .env file.

## Running the Application
To start the application, simply run the `main.py` script:
```bash
python main.py
```

Ensure your environment is set up for audio and visual inputs.

## License
This project is licensed under the MIT License.
