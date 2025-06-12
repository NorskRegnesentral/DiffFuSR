#!/bin/bash

# Set up Python virtual environment
echo "Setting up virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To activate the virtual environment, run: source venv/bin/activate"
