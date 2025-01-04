#!/bin/bash

echo "Starting the environment setup..."
python -m venv senv
source senv/bin/activate

echo "Installing dependencies..."
pip install numpy torch torchvision torchaudio
pip install -r requirements.txt
echo "Setup complete!"
