#!/bin/bash

# This script installs the SARLens package and its dependencies

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment called openmmlab
conda create --name openmmlab python=3.8 -y

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate openmmlab; then
    echo "openmmlab environment activated"
else
    echo "openmmlab environment not found"
    exit 1
fi

# Run the setup.py script
python setup.py develop
