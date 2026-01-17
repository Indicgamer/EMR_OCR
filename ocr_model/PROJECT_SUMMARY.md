# OCR Model Development Project - Summary

## Project Overview

This is a complete OCR (Optical Character Recognition) model development framework for medical documents in an Electronic Medical Record (EMR) system. The framework is optimized for Google Colab training and includes all necessary components for building, training, evaluating, and deploying a custom OCR model.

## What's Included

### 1. Core Model Files (src/)
- **model.py**: CRNN architecture
  - Two model variants: CRNN (full) and OCRModel (lightweight)
  - ResNet-34 feature extraction + Bidirectional LSTM
  - Pre-trained weights support
  
- **dataset.py**: Custom dataset loaders
  - OCRDataset: Generic image-text pair loading
  - MedicalDocumentDataset: Specific to medical docs
  - Data augmentation pipeline
  - Variable-length sequence handling
  
- **utils.py**: Utility functions
  - Character encoding/decoding
  - Metrics (CER, WER calculation)
  - Checkpoint management
  - Visualization functions
  
- **train.py**: Training script template
  - Modular training architecture
  - Logging and checkpointing
  
- **inference.py**: Inference engine
  - Model loading and prediction
  - Image preprocessing

### 2. Data Processing (data_processing/)
- **prepare_dataset.py**: Dataset preparation
  - Loads prescription images and texts
  - Loads lab report images and texts
  - Creates train/val/test splits
  - Generates manifest files
  
- **extract_images.py**: PDF extraction (optional)
  - Converts PDF pages to images
  - Batch processing

### 3. Google Colab Notebooks (notebooks/)
- **01_data_exploration.ipynb**: Data analysis
  - Load and visualize data
  - Dataset statistics
  - Text length distribution
  - Character set analysis
  
- **03_model_training.ipynb**: Main training pipeline
  - Complete end-to-end training
  - Model setup and configuration
  - Training loop with early stopping
  - Evaluation and inference
  - Results visualization

### 4. Configuration (configs/)
- **config.yaml**: Centralized configuration
  - Dataset paths and preprocessing
  - Model hyperparameters
  - Training settings
  - Augmentation parameters

### 5. Documentation
- **README.md**: Comprehensive guide
- **QUICKSTART.md**: Quick start for Google Colab
- **SETUP.md**: Detailed installation instructions

## Dataset

### Prescriptions (data/data1/)
- 130 text files containing prescription OCR output
- Each text file: `{number}.txt` (1.txt to 130.txt)
- Location: `data/data1/Output/`

### Lab Reports (data/lbmaske/)
- Multiple PDF page extractions as text
- Format: `{hospital_code}_{patient_id}_{filename}.txt`
- ~500+ text files
- Location: `data/lbmaske/Output/`

### Data Format
- Text files: UTF-8 encoded plain text
- Image files (optional input): PNG, JPG, BMP, TIFF
- Images: Resized to 64×256 pixels for training

## Model Architecture

### CRNN (Convolutional Recurrent Neural Network)

```
Input Layer (RGB Image 64×256)
    ↓
Convolutional Layers (ResNet-34 backbone)
- Feature extraction: 3 → 512 channels
- Spatial reduction through pooling
    ↓
Bidirectional LSTM (512 → 256 hidden)
- Sequence modeling
- Captures text patterns
    ↓
Dense Layer (256 × 2 → num_classes)
- Character-level predictions
    ↓
CTC Loss
- Handles variable-length sequences
    ↓
Output: Character sequences (text)
```

### Key Features
- **CNN Backbone**: ResNet-34 pre-trained
- **Sequence Modeling**: 2-layer Bidirectional LSTM
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Total Parameters**: ~25-30M
- **Model Size**: ~50MB

## Training Pipeline

### 1. Data Preparation
```
Raw Data (images + text files)
    ↓
prepare_dataset.py
    ↓
Train (80%) / Val (10%) / Test (10%)
    ↓
Data augmentation applied to training set
```

### 2. Training Configuration
- **Batch Size**: 32 (adjustable for GPU memory)
- **Epochs**: 30-50
- **Learning Rate**: 1e-4 (cosine annealing)
- **Optimizer**: Adam with weight decay
- **Early Stopping**: Patience = 5 epochs

### 3. Augmentation Techniques
- Rotation: ±10°
- Shearing: up to 20%
- Brightness/Contrast: ±30%
- Gaussian Blur: 20% probability
- Gaussian Noise: 10% probability
- Elastic Distortion: 10% probability

### 4. Training Loop
```python
For each epoch:
    1. Train on training set
       - Forward pass
       - Calculate CTC loss
       - Backward pass
       - Update weights
    2. Validate on validation set
    3. Save best model
    4. Update learning rate
    5. Check early stopping
```

## Evaluation Metrics

### Character Error Rate (CER)
- Measures character-level accuracy
- Calculated using Levenshtein distance
- Formula: `errors / total_characters`
- Target: < 5% for medical documents

### Word Error Rate (WER)
- Measures word-level accuracy
- Also uses edit distance
- Formula: `word_errors / total_words`

### Training Loss
- CTC loss value
- Should decrease over epochs
- Indicates convergence

## Usage Examples

### Google Colab (Recommended)
1. Upload project to Google Drive
2. Open `notebooks/03_model_training.ipynb` in Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run cells sequentially
5. Monitor training progress
6. Download trained model from checkpoints/

### Local Machine
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Prepare data
python data_processing/prepare_dataset.py

# Train model
python src/train.py --epochs 50 --batch-size 32

# Make predictions
python src/inference.py --model-path checkpoints/best_model.pth --image-path test.png
```

## Performance Expectations

### Training Time
- Google Colab GPU: 2-4 hours for 50 epochs
- Local CPU: 12-24 hours for 50 epochs
- Local GPU: 1-2 hours for 50 epochs

### Accuracy
- Character Error Rate: 3-8% (depends on data quality)
- Inference speed: ~100 images/minute on GPU

### Model Size
- Trained model: ~50MB
- Can be quantized to ~15MB

## File Organization

```
ocr_model/
├── src/
│   ├── model.py                    # CRNN model
│   ├── dataset.py                  # Dataset classes
│   ├── utils.py                    # Helper functions
│   ├── train.py                    # Training script
│   ├── inference.py                # Inference engine
│   └── __init__.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data analysis
│   └── 03_model_training.ipynb     # Training pipeline
│
├── data_processing/
│   ├── prepare_dataset.py          # Dataset prep
│   └── extract_images.py           # PDF extraction
│
├── configs/
│   └── config.yaml                 # Configuration
│
├── checkpoints/                     # Saved models
│   ├── best_model.pth
│   └── training_history.json
│
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
└── SETUP.md                        # Installation guide
```

## Technology Stack

### Deep Learning
- PyTorch 2.0+: Deep learning framework
- TorchVision: Pre-trained models

### Data Processing
- OpenCV: Image processing
- Pillow: Image library
- NumPy: Numerical computing
- Pandas: Data manipulation

### Utilities
- Albumentations: Data augmentation
- scikit-learn: ML utilities
- Matplotlib: Visualization

## Next Steps After Training

### 1. Evaluate Model
- Check Character Error Rate
- Review predictions on test set
- Analyze error patterns

### 2. Improve Performance
- Add more training data
- Fine-tune hyperparameters
- Use ensemble methods

### 3. Deploy Model
- Use `src/inference.py` for predictions
- Integrate with EMR backend
- Create API endpoint

### 4. Monitor and Update
- Track OCR errors in production
- Collect correction feedback
- Retrain periodically

## Troubleshooting Guide

### Common Issues

1. **Out of Memory Error**
   - Reduce batch_size to 16
   - Use CPU mode for debugging
   
2. **Slow Training**
   - Enable GPU: Runtime → Change runtime type
   - Reduce num_workers to 0
   
3. **Poor Accuracy**
   - Train for more epochs
   - Check data preprocessing
   - Verify data quality
   
4. **Model Not Converging**
   - Adjust learning rate
   - Check data augmentation
   - Verify loss calculation

## References & Research

### Papers
- CRNN: https://arxiv.org/abs/1507.05717
- CTC Loss: https://arxiv.org/abs/1311.2508
- Medical OCR: https://arxiv.org/abs/1907.09893

### Libraries
- PyTorch: https://pytorch.org
- Albumentations: https://albumentations.ai
- OpenCV: https://opencv.org

## Key Components Explanation

### Why CRNN?
- Proven architecture for OCR tasks
- Efficient feature extraction via CNN
- Handles variable-length text via LSTM
- CTC loss allows alignment-free training

### Why CTC Loss?
- Doesn't require character-level alignment
- Handles collapsed predictions
- Works with variable-length outputs
- Standard for sequence tasks

### Why Data Augmentation?
- Improves generalization
- Simulates document variations
- Medical documents vary in:
  - Orientation
  - Lighting
  - Image quality
  - Handwriting styles

## Success Criteria

✓ Dataset preparation completes without errors
✓ Model trains without memory issues
✓ Training loss decreases over epochs
✓ Validation loss plateaus at reasonable value
✓ Character Error Rate < 10%
✓ Can make predictions on new images
✓ Model saves successfully

## Support & Debugging

For issues:
1. Check SETUP.md for installation
2. Review QUICKSTART.md for training
3. Check notebook outputs for errors
4. Adjust hyperparameters in config.yaml
5. Enable debug logging for more info

## Future Enhancements

- Multi-language support
- Real-time training visualization
- Model ensemble methods
- Transfer learning from English models
- ONNX model export
- REST API for deployment
- Web interface for predictions

---

**Status**: Complete and ready for Google Colab training
**Last Updated**: January 2026
**Version**: 1.0
