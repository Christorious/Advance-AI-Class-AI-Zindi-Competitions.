"""
License Plate Recognition Module
Uses CNN + LSTM (CRNN) for recognizing characters in license plate images
Tunisian license plates have format: XX-XXX-XX or similar (7 digits total)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for license plate character recognition
    Architecture: CNN feature extractor + Bidirectional LSTM + CTC/Classification head
    """
    
    def __init__(self, num_classes=10, rnn_hidden_size=256, num_rnn_layers=2):
        super(CRNN, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate output size after CNN
        # Input: (batch, 3, 64, 256) -> After CNN: (batch, 512, 4, 64)
        self.cnn_output_channels = 512
        self.cnn_output_height = 4
        self.cnn_output_width = 64
        
        # RNN - Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_channels * self.cnn_output_height,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_rnn_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)
        
        self.num_classes = num_classes
        self.max_seq_length = 7  # Tunisian plates have 7 digits
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        
        Returns:
            logits: Per-character class probabilities (batch, seq_len, num_classes)
        """
        # CNN feature extraction
        features = self.cnn(x)  # (batch, 512, 4, 64)
        
        # Reshape for RNN: (batch, width, channels * height)
        batch_size = features.size(0)
        features = features.permute(0, 3, 1, 2)  # (batch, 64, 512, 4)
        features = features.contiguous().view(batch_size, -1, self.cnn_output_channels * self.cnn_output_height)
        # (batch, 64, 2048)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(features)  # (batch, 64, hidden*2)
        
        # Classification
        logits = self.classifier(rnn_out)  # (batch, 64, num_classes)
        
        return logits
    
    def predict(self, x):
        """
        Predict characters from input image
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        
        Returns:
            predictions: List of predicted digit sequences
            confidences: Confidence scores for each prediction
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, seq_len, num_classes)
            
            # Get predictions using argmax
            probs = F.softmax(logits, dim=-1)
            predictions = []
            confidences = []
            
            for batch_idx in range(logits.size(0)):
                batch_probs = probs[batch_idx]  # (seq_len, num_classes)
                
                # For each of the 7 positions, get the most likely digit
                pred_digits = []
                pred_confidences = []
                
                for pos in range(self.max_seq_length):
                    pos_probs = batch_probs[pos]
                    pred_digit = pos_probs.argmax().item()
                    confidence = pos_probs[pred_digit].item()
                    
                    pred_digits.append(pred_digit)
                    pred_confidences.append(confidence)
                
                predictions.append(pred_digits)
                confidences.append(np.mean(pred_confidences))
        
        return predictions, confidences


def load_recognition_annotations(csv_path: str) -> dict:
    """
    Load recognition annotations from CSV file
    
    Expected columns: image_name, label (the 7-digit plate number)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    annotations = {}
    
    for _, row in df.iterrows():
        img_name = row['image_name'] if 'image_name' in row.columns else row.iloc[0]
        
        # Get label column
        cols = df.columns.tolist()
        label_col = 'label' if 'label' in cols else cols[1]
        
        label = str(row[label_col])
        # Ensure label is exactly 7 digits (pad with leading zeros if needed)
        label = label.zfill(7)[-7:]  # Take last 7 digits if longer
        
        annotations[img_name] = label
    
    return annotations


def preprocess_image_for_recognition(image: np.ndarray, input_size: Tuple[int, int] = (64, 256)) -> np.ndarray:
    """
    Preprocess license plate image for recognition model
    
    Args:
        image: License plate image (BGR format from OpenCV)
        input_size: Target size (height, width)
    
    Returns:
        Preprocessed image tensor ready for model input
    """
    # Convert to grayscale if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Resize to fixed size
    image_resized = cv2.resize(image, input_size)
    
    # Apply preprocessing for better OCR
    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to 3-channel
    image_processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Normalize to [0, 1]
    image_normalized = image_processed.astype(np.float32) / 255.0
    
    # Convert to CHW format and normalize with ImageNet stats (optional)
    image_tensor = np.transpose(image_normalized, (2, 0, 1))
    
    # Simple normalization
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor


def train_recognition_model(
    train_csv: str,
    train_images_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the license plate recognition model
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    
    # Load annotations
    annotations = load_recognition_annotations(train_csv)
    image_names = list(annotations.keys())
    
    class RecognitionDataset(Dataset):
        def __init__(self, image_names, annotations, images_dir, input_size=(64, 256)):
            self.image_names = image_names
            self.annotations = annotations
            self.images_dir = images_dir
            self.input_size = input_size
            self.max_seq_length = 7
            
        def __len__(self):
            return len(self.image_names)
        
        def __getitem__(self, idx):
            img_name = self.image_names[idx]
            img_path = os.path.join(self.images_dir, img_name)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                # Create blank image if loading fails
                image = np.zeros((64, 256, 3), dtype=np.uint8)
            
            # Preprocess
            image_tensor = preprocess_image_for_recognition(image, self.input_size)
            
            # Get target label
            label_str = self.annotations[img_name]
            # Convert string label to list of integers
            label = [int(c) for c in label_str[:self.max_seq_length]]
            # Pad if necessary
            while len(label) < self.max_seq_length:
                label.insert(0, 0)  # Pad with zeros at the beginning
            
            return torch.FloatTensor(image_tensor), torch.LongTensor(label)
    
    # Create dataset and dataloader
    dataset = RecognitionDataset(image_names, annotations, train_images_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = CRNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)  # (batch, 7)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)  # (batch, 64, 10)
            
            # We only care about first 7 positions
            logits_trimmed = logits[:, :7, :]  # (batch, 7, 10)
            
            # Reshape for loss calculation
            logits_flat = logits_trimmed.view(-1, 10)  # (batch*7, 10)
            targets_flat = targets.view(-1)  # (batch*7,)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = logits_trimmed.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, 'best_recognition_model.pth'))
    
    print(f"Training completed. Best loss: {best_loss:.4f}, Best accuracy: {accuracy:.4f}")
    return model


def create_submission_format(predictions: List[List[int]], image_names: List[str]) -> pd.DataFrame:
    """
    Create submission file in the required format
    
    Each image needs 7 rows (one for each digit position)
    Each row is one-hot encoded for digits 0-9
    
    Args:
        predictions: List of predicted digit sequences (each sequence has 7 digits)
        image_names: List of image names
    
    Returns:
        DataFrame ready for submission
    """
    import pandas as pd
    
    rows = []
    
    for img_name, pred_digits in zip(image_names, predictions):
        # Extract base name without extension
        base_name = os.path.splitext(img_name)[0]
        
        for pos, digit in enumerate(pred_digits, start=1):
            row_id = f"{base_name}_{pos}"
            
            # One-hot encode the digit
            one_hot = [0] * 10
            one_hot[digit] = 1
            
            rows.append({
                'id': row_id,
                **{str(i): one_hot[i] for i in range(10)}
            })
    
    # Create DataFrame
    columns = ['id'] + [str(i) for i in range(10)]
    df = pd.DataFrame(rows, columns=columns)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("License Plate Recognition Module")
    print("Use train_recognition_model() to train the recognizer")
    print("Use CRNN class for inference")
