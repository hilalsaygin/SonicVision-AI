
echo "Starting the environment setup..."
python -m venv sv-env
echo "Activating the virtual environment..."
source sv-env/bin/activate

echo "Installing dependencies..."
pip install numpy torch torchvision torchaudio
pip install -r requirements.txt
