"""
Main Pipeline for Tunisian License Plate Detection and Recognition

This script combines detection and recognition models to:
1. Detect license plates in vehicle images
2. Extract the license plate region
3. Recognize the characters on the license plate
4. Generate submission file in the required format
"""

import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# Import our modules
from license_plate_detection import (
    LicensePlateDetector, 
    preprocess_image_for_detection, 
    postprocess_bbox,
    extract_license_plate,
    load_detection_annotations
)
from license_plate_recognition import (
    CRNN, 
    preprocess_image_for_recognition,
    load_recognition_annotations,
    create_submission_format
)


class LicensePlatePipeline:
    """
    End-to-end pipeline for license plate detection and recognition
    """
    
    def __init__(
        self,
        detection_model_path: str = None,
        recognition_model_path: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Initialize detection model
        self.detection_model = LicensePlateDetector().to(device)
        if detection_model_path and os.path.exists(detection_model_path):
            checkpoint = torch.load(detection_model_path, map_location=device)
            self.detection_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded detection model from {detection_model_path}")
        else:
            print("Warning: No detection model loaded. Using random weights.")
        
        # Initialize recognition model
        self.recognition_model = CRNN(num_classes=10).to(device)
        if recognition_model_path and os.path.exists(recognition_model_path):
            checkpoint = torch.load(recognition_model_path, map_location=device)
            self.recognition_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded recognition model from {recognition_model_path}")
        else:
            print("Warning: No recognition model loaded. Using random weights.")
        
        self.detection_model.eval()
        self.recognition_model.eval()
    
    def detect_license_plate(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Detect license plate in a vehicle image
        
        Returns:
            bbox: Bounding box [ymin, xmin, ymax, xmax] in original image coordinates
            metadata: Preprocessing metadata
        """
        # Preprocess image
        image_tensor, metadata = preprocess_image_for_detection(image_path)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(image_tensor).unsqueeze(0).to(self.device)
        
        # Run detection
        with torch.no_grad():
            bbox_normalized = self.detection_model(input_tensor)
            bbox_normalized = bbox_normalized.cpu().numpy()[0]
        
        # Postprocess to get original coordinates
        bbox_original = postprocess_bbox(bbox_normalized, metadata)
        
        return bbox_original, metadata
    
    def recognize_license_plate(self, plate_image: np.ndarray) -> List[int]:
        """
        Recognize characters in a license plate image
        
        Returns:
            pred_digits: List of 7 predicted digits
        """
        # Preprocess plate image
        plate_tensor = preprocess_image_for_recognition(plate_image)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(plate_tensor).unsqueeze(0).to(self.device)
        
        # Run recognition
        predictions, confidences = self.recognition_model.predict(input_tensor)
        
        return predictions[0], confidences[0]
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single vehicle image through the full pipeline
        
        Returns:
            Dictionary containing:
                - bbox: Detected bounding box
                - plate_image: Extracted license plate image
                - prediction: List of 7 predicted digits
                - confidence: Average confidence score
        """
        # Step 1: Detect license plate
        bbox, metadata = self.detect_license_plate(image_path)
        
        # Step 2: Extract license plate region
        plate_image = extract_license_plate(image_path, bbox)
        
        # Step 3: Recognize characters
        pred_digits, confidence = self.recognize_license_plate(plate_image)
        
        return {
            'bbox': bbox,
            'plate_image': plate_image,
            'prediction': pred_digits,
            'confidence': confidence
        }
    
    def process_test_set(
        self,
        test_images_dir: str,
        test_csv: str = None,
        output_path: str = 'submission.csv'
    ) -> pd.DataFrame:
        """
        Process all images in the test set and generate submission file
        
        Args:
            test_images_dir: Directory containing test images
            test_csv: Optional CSV file with test image names
            output_path: Path to save submission file
        """
        # Get list of test images
        if test_csv and os.path.exists(test_csv):
            df_test = pd.read_csv(test_csv)
            image_names = df_test['image_name'].tolist() if 'image_name' in df_test.columns else df_test.iloc[:, 0].tolist()
        else:
            # Scan directory for images
            image_names = sorted([
                f for f in os.listdir(test_images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        
        all_predictions = []
        processed_names = []
        
        print(f"Processing {len(image_names)} test images...")
        
        for i, img_name in enumerate(image_names):
            img_path = os.path.join(test_images_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            try:
                result = self.process_image(img_path)
                all_predictions.append(result['prediction'])
                processed_names.append(img_name)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_names)} images")
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                # Use default prediction (all zeros) for failed images
                all_predictions.append([0] * 7)
                processed_names.append(img_name)
        
        # Create submission file
        submission_df = create_submission_format(all_predictions, processed_names)
        
        # Save submission
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")
        
        return submission_df


def train_full_pipeline(
    detection_train_csv: str,
    detection_images_dir: str,
    recognition_train_csv: str,
    recognition_images_dir: str,
    output_dir: str,
    detection_epochs: int = 50,
    recognition_epochs: int = 50
):
    """
    Train both detection and recognition models
    """
    print("=" * 60)
    print("Training License Plate Detection Model")
    print("=" * 60)
    
    from license_plate_detection import train_detection_model
    
    detection_output = os.path.join(output_dir, 'detection')
    train_detection_model(
        train_csv=detection_train_csv,
        train_images_dir=detection_images_dir,
        output_dir=detection_output,
        epochs=detection_epochs
    )
    
    print("\n" + "=" * 60)
    print("Training License Plate Recognition Model")
    print("=" * 60)
    
    from license_plate_recognition import train_recognition_model
    
    recognition_output = os.path.join(output_dir, 'recognition')
    train_recognition_model(
        train_csv=recognition_train_csv,
        train_images_dir=recognition_images_dir,
        output_dir=recognition_output,
        epochs=recognition_epochs
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Detection model saved to: {os.path.join(detection_output, 'best_detection_model.pth')}")
    print(f"Recognition model saved to: {os.path.join(recognition_output, 'best_recognition_model.pth')}")
    print("=" * 60)


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Tunisian License Plate Detection and Recognition')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict',
                       help='Mode: train or predict')
    parser.add_argument('--detection-train-csv', type=str,
                       help='Path to detection training CSV')
    parser.add_argument('--detection-images-dir', type=str,
                       help='Path to detection training images')
    parser.add_argument('--recognition-train-csv', type=str,
                       help='Path to recognition training CSV')
    parser.add_argument('--recognition-images-dir', type=str,
                       help='Path to recognition training images')
    parser.add_argument('--test-images-dir', type=str,
                       help='Path to test images directory')
    parser.add_argument('--test-csv', type=str,
                       help='Path to test CSV file')
    parser.add_argument('--detection-model', type=str,
                       help='Path to pre-trained detection model')
    parser.add_argument('--recognition-model', type=str,
                       help='Path to pre-trained recognition model')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for trained models')
    parser.add_argument('--output-file', type=str, default='submission.csv',
                       help='Output path for submission file')
    parser.add_argument('--detection-epochs', type=int, default=50,
                       help='Number of epochs for detection training')
    parser.add_argument('--recognition-epochs', type=int, default=50,
                       help='Number of epochs for recognition training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not all([args.detection_train_csv, args.detection_images_dir,
                   args.recognition_train_csv, args.recognition_images_dir]):
            print("Error: Training mode requires all training data paths")
            print("Required arguments:")
            print("  --detection-train-csv")
            print("  --detection-images-dir")
            print("  --recognition-train-csv")
            print("  --recognition-images-dir")
            sys.exit(1)
        
        train_full_pipeline(
            detection_train_csv=args.detection_train_csv,
            detection_images_dir=args.detection_images_dir,
            recognition_train_csv=args.recognition_train_csv,
            recognition_images_dir=args.recognition_images_dir,
            output_dir=args.output_dir,
            detection_epochs=args.detection_epochs,
            recognition_epochs=args.recognition_epochs
        )
    
    elif args.mode == 'predict':
        if not args.test_images_dir:
            print("Error: Prediction mode requires test images directory")
            print("Required argument: --test-images-dir")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = LicensePlatePipeline(
            detection_model_path=args.detection_model,
            recognition_model_path=args.recognition_model
        )
        
        # Process test set
        pipeline.process_test_set(
            test_images_dir=args.test_images_dir,
            test_csv=args.test_csv,
            output_path=args.output_file
        )
        
        print(f"\nPrediction complete! Submission saved to: {args.output_file}")


if __name__ == "__main__":
    main()
