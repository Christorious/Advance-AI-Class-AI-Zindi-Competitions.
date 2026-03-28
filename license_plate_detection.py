"""
License Plate Detection Module
Uses YOLOv8 or custom CNN for detecting license plates in vehicle images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Optional
import os


class LicensePlateDetector(nn.Module):
    """
    Custom CNN for license plate detection
    Outputs bounding box coordinates: [ymin, xmin, ymax, xmax]
    """
    
    def __init__(self, num_classes=1, pretrained_backbone=False):
        super(LicensePlateDetector, self).__init__()
        
        # Backbone - Feature extraction
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # Output: ymin, xmin, ymax, xmax
        )
        
    def forward(self, x):
        features = self.backbone(x)
        bbox = self.detector(features)
        return bbox


def load_detection_annotations(csv_path: str) -> dict:
    """
    Load detection annotations from CSV file
    
    Expected columns: image_name, ymin, xmin, ymax, xmax
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    annotations = {}
    
    for _, row in df.iterrows():
        img_name = row['image_name'] if 'image_name' in row.columns else row.iloc[0]
        
        # Get column names dynamically
        cols = df.columns.tolist()
        ymin_col = 'ymin' if 'ymin' in cols else cols[1]
        xmin_col = 'xmin' if 'xmin' in cols else cols[2]
        ymax_col = 'ymax' if 'ymax' in cols else cols[3]
        xmax_col = 'xmax' if 'xmax' in cols else cols[4]
        
        annotations[img_name] = {
            'ymin': float(row[ymin_col]),
            'xmin': float(row[xmin_col]),
            'ymax': float(row[ymax_col]),
            'xmax': float(row[xmax_col])
        }
    
    return annotations


def preprocess_image_for_detection(image_path: str, input_size: Tuple[int, int] = (416, 416)) -> Tuple[np.ndarray, dict]:
    """
    Preprocess image for detection model
    
    Returns:
        - Resized image tensor
        - Metadata for post-processing (scale factors, padding)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]  # (height, width)
    
    # Resize image
    image_resized = cv2.resize(image, input_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to CHW format for PyTorch
    image_tensor = np.transpose(image_normalized, (2, 0, 1))
    
    metadata = {
        'original_shape': original_shape,
        'input_size': input_size,
        'scale_y': original_shape[0] / input_size[0],
        'scale_x': original_shape[1] / input_size[1]
    }
    
    return image_tensor, metadata


def postprocess_bbox(bbox: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Convert predicted bbox back to original image coordinates
    
    bbox: [ymin, xmin, ymax, xmax] in normalized/resized coordinates
    """
    scale_y = metadata['scale_y']
    scale_x = metadata['scale_x']
    
    bbox_original = bbox.copy()
    bbox_original[0] *= scale_y  # ymin
    bbox_original[1] *= scale_x  # xmin
    bbox_original[2] *= scale_y  # ymax
    bbox_original[3] *= scale_x  # xmax
    
    # Clip to image boundaries
    h, w = metadata['original_shape']
    bbox_original[0] = max(0, min(bbox_original[0], h))
    bbox_original[1] = max(0, min(bbox_original[1], w))
    bbox_original[2] = max(0, min(bbox_original[2], h))
    bbox_original[3] = max(0, min(bbox_original[3], w))
    
    return bbox_original


def extract_license_plate(image_path: str, bbox: np.ndarray) -> np.ndarray:
    """
    Extract license plate region from image using bounding box
    
    bbox: [ymin, xmin, ymax, xmax] in original image coordinates
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    ymin, xmin, ymax, xmax = bbox.astype(int)
    
    # Add small margin for better recognition
    margin_y = int((ymax - ymin) * 0.1)
    margin_x = int((xmax - xmin) * 0.1)
    
    ymin = max(0, ymin - margin_y)
    xmin = max(0, xmin - margin_x)
    ymax = min(image.shape[0], ymax + margin_y)
    xmax = min(image.shape[1], xmax + margin_x)
    
    plate_image = image[ymin:ymax, xmin:xmax]
    
    return plate_image


def train_detection_model(
    train_csv: str,
    train_images_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the license plate detection model
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    
    # Load annotations
    annotations = load_detection_annotations(train_csv)
    image_names = list(annotations.keys())
    
    class DetectionDataset(Dataset):
        def __init__(self, image_names, annotations, images_dir, input_size=(416, 416)):
            self.image_names = image_names
            self.annotations = annotations
            self.images_dir = images_dir
            self.input_size = input_size
            
        def __len__(self):
            return len(self.image_names)
        
        def __getitem__(self, idx):
            img_name = self.image_names[idx]
            img_path = os.path.join(self.images_dir, img_name)
            
            # Load and preprocess image
            image = cv2.imread(img_path)
            image_resized = cv2.resize(image, self.input_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            
            # Get target bbox
            ann = self.annotations[img_name]
            bbox = np.array([ann['ymin'], ann['xmin'], ann['ymax'], ann['xmax']], dtype=np.float32)
            
            # Normalize bbox to [0, 1] based on input size
            bbox[0] /= image.shape[0]  # ymin
            bbox[1] /= image.shape[1]  # xmin
            bbox[2] /= image.shape[0]  # ymax
            bbox[3] /= image.shape[1]  # xmax
            
            return torch.FloatTensor(image_tensor), torch.FloatTensor(bbox)
    
    # Create dataset and dataloader
    dataset = DetectionDataset(image_names, annotations, train_images_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = LicensePlateDetector().to(device)
    criterion = nn.SmoothL1Loss()  # Huber loss for bbox regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, 'best_detection_model.pth'))
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    # Example usage
    print("License Plate Detection Module")
    print("Use train_detection_model() to train the detector")
    print("Use LicensePlateDetector class for inference")
