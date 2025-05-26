import os
import sys
import requests
import tensorflow as tf
from tqdm import tqdm
import zipfile
import shutil

def download_file(url, filename):
    """
    Download a file from a URL with a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def extract_dataset(zip_path, extract_path):
    """
    Extract the dataset zip file
    """
    print(f"Extracting dataset to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            zip_ref.extract(member, extract_path)
    print("Dataset extracted successfully!")

def main():
    """
    Download and prepare the model and dataset
    """
    print("Setting up the Plant Disease Detection model...")
    
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)
    
    # Instructions for dataset download
    print("\n======= DATASET DOWNLOAD INSTRUCTIONS =======")
    print("1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("2. Click the 'Download' button")
    print("3. Save the zip file to the project directory")
    print("4. Enter the path to the downloaded zip file below (or press Enter to skip if already extracted)")
    
    # Ask for the dataset zip path
    dataset_zip_path = input("\nEnter path to downloaded dataset zip (or press Enter to skip): ").strip()
    
    if dataset_zip_path and os.path.exists(dataset_zip_path):
        # Extract the dataset
        extract_path = input("Enter extraction path (or press Enter for current directory): ").strip()
        if not extract_path:
            extract_path = '.'
        
        extract_dataset(dataset_zip_path, extract_path)
        
        print("\nDataset successfully prepared!")
        print("The dataset should have the following structure:")
        print("- train/    (containing training images)")
        print("- valid/    (containing validation images)")
        print("- test/     (containing test images)")
    else:
        print("\nSkipping dataset extraction. Make sure you have the dataset properly set up.")
    
    # Check if model already exists
    if os.path.exists('trained_plant_disease_model.keras'):
        print("\nPre-trained model already exists. Skipping download.")
    else:
        print("\nNo pre-trained model found. You have two options:")
        print("1. Train the model using the dataset (this may take several hours)")
        print("2. Download a pre-trained model from Kaggle or other source")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\nPlease run the training script with:")
            print("python train_model.py")
            print("Note: Training will take several hours depending on your hardware.")
            
        elif choice == '2':
            print("\nTo download a pre-trained model:")
            print("1. Look for plant disease detection models on Kaggle")
            print("2. Download the model file")
            print("3. Rename it to 'trained_plant_disease_model.keras'")
            print("4. Place it in the project root directory")
            
            model_path = input("\nEnter path to downloaded model file (or press Enter to skip): ").strip()
            if model_path and os.path.exists(model_path):
                shutil.copy(model_path, 'trained_plant_disease_model.keras')
                print("Model copied successfully to project directory!")
            else:
                print("No model provided. You'll need to provide a model before running the application.")
    
    print("\nSetup complete! When ready, run 'python app.py' to start the application.")

if __name__ == "__main__":
    main() 