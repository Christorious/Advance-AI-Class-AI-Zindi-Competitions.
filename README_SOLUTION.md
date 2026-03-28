# Tunisian License Plate Detection and Recognition

This project provides a complete computer vision solution for detecting and recognizing Tunisian vehicle license plates. It was designed for the AI Tunisia Hack 2019 competition.

## Overview

The solution consists of two main components:

1. **License Plate Detection**: A CNN-based model that detects and localizes license plates in vehicle images
2. **License Plate Recognition**: A CRNN (Convolutional Recurrent Neural Network) that recognizes the 7 digits on Tunisian license plates

## Project Structure

```
/workspace/
├── license_plate_detection.py    # Detection model and utilities
├── license_plate_recognition.py  # Recognition model and utilities
├── main_pipeline.py              # End-to-end pipeline
├── train_solution.py             # Training script with example usage
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- OpenCV (opencv-python-headless)
- NumPy
- Pandas

Install dependencies:
```bash
pip install torch torchvision opencv-python-headless numpy pandas
```

## Data Format

### Detection Training Data
- CSV with columns: `image_name`, `ymin`, `xmin`, `ymax`, `xmax`
- Images: Vehicle images containing license plates

### Recognition Training Data
- CSV with columns: `image_name`, `label` (7-digit plate number)
- Images: Cropped license plate images

### Test Data
- Vehicle images (similar to detection training data)
- Output: 7-digit predictions for each image

## Usage

### Training

To train both models:

```bash
python main_pipeline.py --mode train \
    --detection-train-csv path/to/detection_train.csv \
    --detection-images-dir path/to/detection_images/ \
    --recognition-train-csv path/to/recognition_train.csv \
    --recognition-images-dir path/to/recognition_images/ \
    --output-dir ./models \
    --detection-epochs 50 \
    --recognition-epochs 50
```

Or use the provided training script:

```bash
python train_solution.py
```

### Prediction / Inference

To generate predictions on test data:

```bash
python main_pipeline.py --mode predict \
    --test-images-dir path/to/test_images/ \
    --test-csv path/to/test.csv \
    --detection-model ./models/detection/best_detection_model.pth \
    --recognition-model ./models/recognition/best_recognition_model.pth \
    --output-file submission.csv
```

## Output Format

The submission file follows this format:
- Each image generates 7 rows (one for each digit position)
- Each row contains one-hot encoded probabilities for digits 0-9
- Row ID format: `{image_name}_{position}` (e.g., `img_1_1`, `img_1_2`, ..., `img_1_7`)

Example:
```csv
id,0,1,2,3,4,5,6,7,8,9
img_1_1,0,1,0,0,0,0,0,0,0,0
img_1_2,0,0,0,0,0,0,1,0,0,0
...
img_1_7,0,0,0,0,0,1,0,0,0,0
```

## Model Architecture

### Detection Model
- Backbone: Custom CNN with 4 convolutional blocks
- Head: Fully connected layers outputting 4 coordinates (ymin, xmin, ymax, xmax)
- Loss: SmoothL1Loss (Huber loss) for bounding box regression

### Recognition Model (CRNN)
- CNN: 5 convolutional blocks for feature extraction
- RNN: Bidirectional LSTM for sequence modeling
- Classifier: Linear layer for digit classification (10 classes)
- Loss: CrossEntropyLoss for multi-class classification

## Evaluation Metric

The competition uses **Log Loss** (binary cross-entropy) as the evaluation metric. Each digit position is evaluated independently, and the final score is the average log loss across all positions and all images.

## Tunisian License Plate Format

Tunisian license plates follow this format:
- Total: 7 digits
- First part: 2-3 digits
- Second part: 4-5 digits
- Example: `12-345-67` or `123-4567`

For submission, if the first number has fewer than 3 digits, it should be padded with leading zeros.

## Tips for Better Performance

1. **Data Augmentation**: Apply random rotations, flips, and color jittering during training
2. **Preprocessing**: Use adaptive thresholding and contrast enhancement for recognition
3. **Model Fine-tuning**: Consider using pretrained backbones (ResNet, EfficientNet)
4. **Ensemble**: Combine predictions from multiple models for better accuracy
5. **Post-processing**: Apply constraints based on Tunisian plate format rules

## License

This project was created for educational purposes as part of the AI Tunisia Hack 2019 competition.

## Credits

- Challenge designed by InstaDeep Tunisia
- In partnership with the National Road Safety Observatory of Tunisia
- For more information: http://www.onsr.tn/ and https://www.instadeep.com/
