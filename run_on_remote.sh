#!/bin/bash
# Script to run PredNet training on remote GPU server
# Usage: ./run_on_remote.sh username@remote-server

if [ -z "$1" ]; then
    echo "Usage: ./run_on_remote.sh username@remote-server"
    exit 1
fi

REMOTE=$1
REMOTE_DIR="~/prednet_training"

echo "Transferring script to remote server..."
scp train_prednet_moving_mnist.py ${REMOTE}:${REMOTE_DIR}/

echo "Setting up and running on remote server..."
ssh ${REMOTE} << 'ENDSSH'
cd ~/prednet_training

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade required packages
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision numpy matplotlib tqdm

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run training in screen session
echo "Starting training in screen session..."
screen -dmS prednet_training bash -c "python train_prednet_moving_mnist.py 2>&1 | tee training.log"

echo "Training started in screen session 'prednet_training'"
echo "To attach: ssh ${REMOTE} && screen -r prednet_training"
echo "To check logs: ssh ${REMOTE} && tail -f ~/prednet_training/training.log"
ENDSSH

echo "Done! Training is running on remote server."
echo "To monitor: ssh ${REMOTE} && screen -r prednet_training"






