#!/bin/bash

# This script installs the required Python packages for the project.
# It attempts to detect an NVIDIA GPU and install the appropriate
# version of PyTorch with CUDA support.

echo "--- Starting Project Installation ---"

# --- 1. Create and Activate Virtual Environment ---
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Activating..."
else
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Please ensure python3 and venv are installed."
        exit 1
    fi
fi

source venv/bin/activate
echo "Virtual environment activated."
echo

# --- 2. Install Standard Packages ---
echo "Installing standard packages (ultralytics, opencv, pyyaml)..."
pip install ultralytics opencv-python pyyaml
if [ $? -ne 0 ]; then
    echo "Error: Failed to install standard packages."
    exit 1
fi
echo "Standard packages installed successfully."
echo

# --- 3. Install PyTorch (Detecting GPU) ---
echo "Detecting NVIDIA GPU for PyTorch installation..."

# Check if nvidia-smi command exists and is executable
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support."
    # This command installs PyTorch for the latest stable CUDA version.
    # Users with older CUDA versions might need to adjust this.
    # Find the correct command at: https://pytorch.org/get-started/locally/
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyTorch with CUDA. Please visit https://pytorch.org/get-started/locally/ for manual installation instructions."
        exit 1
    fi
    echo "PyTorch with CUDA support installed successfully."
else
    echo "No NVIDIA GPU detected or 'nvidia-smi' not found. Installing CPU version of PyTorch."
    pip install torch torchvision torchaudio
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install CPU version of PyTorch."
        exit 1
    fi
    echo "CPU version of PyTorch installed successfully."
fi

echo
echo "--- Installation Complete! ---"
echo "You can now run the scripts within the activated virtual environment."
echo "To activate it in the future, run: source venv/bin/activate"