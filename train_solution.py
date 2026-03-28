"""
Training Script for Tunisian License Plate Detection and Recognition

This script demonstrates how to train the models with sample data paths.
Update the paths according to your actual data location.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import train_full_pipeline


def main():
    """
    Main training function
    
    Update these paths to match your data location:
    """
    
    # === UPDATE THESE PATHS TO YOUR DATA ===
    
    # Detection training data
    detection_train_csv = 'path/to/license_plates_detection_train.csv'
    detection_images_dir = 'path/to/detection_images/'  # Folder with vehicle images
    
    # Recognition training data  
    recognition_train_csv = 'path/to/license_plates_recognition_train.csv'
    recognition_images_dir = 'path/to/recognition_images/'  # Folder with plate images
    
    # Output directory for trained models
    output_dir = './models'
    
    # Number of training epochs (reduce for testing, increase for better results)
    detection_epochs = 50
    recognition_epochs = 50
    
    # =======================================
    
    # Check if paths exist (skip this if you haven't downloaded data yet)
    print("Checking data paths...")
    print(f"Detection CSV: {detection_train_csv} - Exists: {os.path.exists(detection_train_csv)}")
    print(f"Detection Images: {detection_images_dir} - Exists: {os.path.exists(detection_images_dir)}")
    print(f"Recognition CSV: {recognition_train_csv} - Exists: {os.path.exists(recognition_train_csv)}")
    print(f"Recognition Images: {recognition_images_dir} - Exists: {os.path.exists(recognition_images_dir)}")
    
    print("\n" + "="*60)
    print("Starting Training Pipeline")
    print("="*60 + "\n")
    
    try:
        train_full_pipeline(
            detection_train_csv=detection_train_csv,
            detection_images_dir=detection_images_dir,
            recognition_train_csv=recognition_train_csv,
            recognition_images_dir=recognition_images_dir,
            output_dir=output_dir,
            detection_epochs=detection_epochs,
            recognition_epochs=recognition_epochs
        )
        
        print("\n" + "="*60)
        print("SUCCESS! Training completed.")
        print("="*60)
        print(f"\nTrained models saved to:")
        print(f"  Detection: {os.path.join(output_dir, 'detection', 'best_detection_model.pth')}")
        print(f"  Recognition: {os.path.join(output_dir, 'recognition', 'best_recognition_model.pth')}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found - {e}")
        print("\nPlease update the paths in this script to point to your actual data files.")
        print("\nTo download the competition data:")
        print("1. Go to: https://zindi.africa/competitions/ai-hack-tunisia-2-computer-vision-challenge-2")
        print("2. Download the training data files:")
        print("   - license_plates_detection_train.zip")
        print("   - license_plates_detection_train.csv")
        print("   - license_plates_recognition_train.zip")
        print("   - license_plates_recognition_train.csv")
        print("3. Extract the zip files")
        print("4. Update the paths in this script")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
