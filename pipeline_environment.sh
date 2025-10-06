#!/bin/bash

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script requires a Linux operating system"
    exit 1
fi

# Print system requirements
echo "System Requirements:"
echo "- Linux operating system"
echo "- NVIDIA GPU with:"
echo "  * Minimum 24GB VRAM for Qwen 7B model"
echo "  * Minimum 40GB VRAM for Qwen 14B model"
echo "- CUDA 12.1 compatible system"
echo "- Conda package manager"
echo

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Conda first."
    exit 1
fi

# Environment setup
CONDA_ENV_NAME="py312torch240cuda121"
echo "Setting up conda environment: $CONDA_ENV_NAME"

# Remove existing environment if it exists
conda deactivate 2>/dev/null
echo "Removing existing environment if it exists..."
conda env remove -n $CONDA_ENV_NAME -y

# Create and activate new environment
echo "Creating new conda environment..."
conda create -n $CONDA_ENV_NAME python=3.12 -y || { echo "Failed to create conda environment"; exit 1; }

# Ensure conda environment is properly activated
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME || { echo "Failed to activate conda environment"; exit 1; }

# Install dependencies
echo "Installing CUDA and PyTorch..."
conda install -y cuda-cudart=12.1.105=0 -c nvidia || { echo "Failed to install CUDA"; exit 1; }
#conda install -y pytorch=2.3.0=py3.12_cuda12.1_cudnn8.9.2_0 -c pytorch || { echo "Failed to install PyTorch"; exit 1; }
conda -y install nvidia/label/cuda-12.1.0::cuda-nvcc
echo "Installing Python packages..."

pip install ninja
#pip install flash-attn --no-build-isolation
pip install modelscope==1.18.0  # For model download
pip install openai==1.46.0
pip install tqdm==4.66.2
pip install transformers==4.44.2
pip install vllm==0.6.1.post2

pip install flash-attn --no-build-isolation


# model download
echo "Downloading Qwen models..."
modelscope download Qwen/Qwen2.5-14B-Instruct
modelscope download Qwen/Qwen2.5-7B-Instruct

echo "Installation completed successfully!"
echo "Activated conda environment: $CONDA_ENV_NAME"
echo "You can now run your scripts using this environment."
