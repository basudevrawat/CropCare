#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_green() {
    echo -e "${GREEN}$1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}$1${NC}"
}

print_blue() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_yellow "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    print_green "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_green "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_green "Installing dependencies..."
pip install -r requirements.txt

# Setup options
print_blue "\n=== Plant Disease Detection Setup ==="
print_blue "1. Download dataset and setup model"
print_blue "2. Train model using existing dataset"
print_blue "3. Skip setup and run application"
print_blue "4. Exit"

read -p "Choose an option (1-4): " option

case $option in
    1)
        # Download and setup
        print_green "\nSetting up dataset and model..."
        python download_model.py
        ;;
    2)
        # Train model
        print_green "\nStarting model training..."
        python train_model.py
        ;;
    3)
        # Skip setup
        print_yellow "\nSkipping setup."
        print_yellow "Make sure you have a trained model file named 'trained_plant_disease_model.keras' in the project directory."
        ;;
    4)
        # Exit
        print_yellow "\nExiting setup."
        exit 0
        ;;
    *)
        print_yellow "\nInvalid option. Exiting."
        exit 1
        ;;
esac

# Ask user if they want to run the application
read -p "Do you want to run the application now? (y/n): " run_app

if [[ $run_app == "y" || $run_app == "Y" ]]; then
    # Run the application
    print_green "\nStarting the application..."
    python app.py
else
    print_green "\nSetup complete. Run 'python app.py' to start the application when ready."
fi 