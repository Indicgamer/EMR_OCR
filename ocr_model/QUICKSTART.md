# OCR Model Development for EMR System - Quick Start Guide

## Overview
This guide helps you quickly set up and train an OCR model for recognizing text from medical documents (prescriptions and lab reports) for your Electronic Medical Record (EMR) system.

## Project Structure
```
ocr_model/
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Explore dataset
│   ├── 03_model_training.ipynb          # Main training notebook
│   ├── 02_data_preprocessing.ipynb      # (Optional) Data prep
│   └── 04_evaluation_inference.ipynb    # (Optional) Evaluation
├── src/
│   ├── model.py                         # Model architecture (CRNN)
│   ├── dataset.py                       # Dataset classes
│   ├── utils.py                         # Utility functions
│   ├── train.py                         # Training script
│   ├── inference.py                     # Inference script
│   └── __init__.py
├── data_processing/
│   ├── prepare_dataset.py               # Dataset preparation
│   └── extract_images.py                # PDF image extraction
├── configs/
│   └── config.yaml                      # Model configuration
├── checkpoints/                         # Saved models
├── requirements.txt                     # Dependencies
└── README.md                            # Detailed documentation
```

## Quick Start - Google Colab

### Step 1: Upload Project to Google Drive
1. Upload the `new_EMR` folder to your Google Drive
2. Note the path: `/MyDrive/new_EMR/`

### Step 2: Open Google Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → Drive tab
3. Navigate to `new_EMR/ocr_model/notebooks/03_model_training.ipynb`

### Step 3: Run Cells Sequentially
Execute notebook cells in order:
- **Cell 1-5**: Setup and installation
- **Cell 6-9**: Data loading and exploration
- **Cell 10-12**: Model architecture setup
- **Cell 13-15**: Training configuration
- **Cell 16-18**: Training loop (monitor GPU usage)
- **Cell 19-21**: Evaluation and results

### Step 4: Monitor Training
- GPU memory usage appears on the right
- Training progress shows in real-time
- Validation loss should decrease over epochs

### Step 5: Download Results
After training:
```
checkpoints/
├── best_model.pth              # Best model weights
└── training_history.json       # Training metrics
```

## Dataset Structure Required
```
data/
├── data1/                       # Prescriptions
│   ├── Input/                   # Prescription images
│   └── Output/                  # Prescription text (1.txt, 2.txt, ...)
└── lbmaske/                     # Lab reports
    ├── Input/                   # Lab report images
    └── Output/                  # Lab report text
```

## Key Configuration Parameters

Edit `configs/config.yaml` to customize:

```yaml
dataset:
  resize_height: 64              # Input image height
  resize_width: 256              # Input image width
  train_split: 0.8               # 80% training

model:
  type: "CRNN"                   # Model architecture
  num_classes: 95                # Character classes
  hidden_size: 256               # LSTM hidden size

training:
  batch_size: 32                 # Batch size (reduce if OOM)
  num_epochs: 50                 # Maximum epochs
  learning_rate: 1e-4            # Learning rate
```

## Common Issues & Solutions

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size in training cell
```python
batch_size = 16  # Instead of 32
```

### Issue: GPU Not Available
**Solution**: Enable GPU in Colab
1. Runtime → Change runtime type
2. Hardware accelerator → GPU

### Issue: Slow Training
**Solution**: 
- Reduce num_workers to 0 or 2
- Use smaller batch size
- Use fewer epochs initially

### Issue: Poor OCR Accuracy
**Solution**:
- Train for more epochs (increase from 30 to 50+)
- Use data augmentation (enabled by default)
- Collect more training data
- Fine-tune learning rate

## Model Architecture

The OCR model uses CRNN (Convolutional Recurrent Neural Network):

```
Input Image (64×256)
    ↓
CNN Feature Extraction (ResNet-34)
    ↓
Bidirectional LSTM (256 hidden units)
    ↓
Dense Layer (character classification)
    ↓
CTC Loss (sequence prediction)
    ↓
Output Text
```

## Training Tips

1. **Data Augmentation**: Automatically applied to training data
   - Rotation: ±10°
   - Shearing: 0-20%
   - Brightness/contrast adjustment
   - Gaussian blur and noise

2. **Learning Rate**: Cosine annealing schedule
   - Starts: 1e-4
   - Ends: 1e-6
   - Warms up for 5 epochs

3. **Early Stopping**: Stops if validation loss doesn't improve for 5 epochs

4. **Loss Function**: CTC (Connectionist Temporal Classification)
   - Suitable for variable-length sequences
   - Handles alignment automatically

## Evaluation Metrics

- **Character Error Rate (CER)**: 
  - Ratio of character-level errors to total characters
  - Target: < 5% for medical documents

- **Training Loss**: Should decrease over epochs
- **Validation Loss**: Should plateau after convergence

## Next Steps After Training

1. **Save Model**:
   ```python
   # Automatically saved to checkpoints/best_model.pth
   ```

2. **Deploy in EMR**:
   - Use `src/inference.py` for predictions
   - Integrate with your EMR backend

3. **Fine-tune on New Data**:
   - Collect OCR errors from production
   - Retrain with corrected labels
   - Improve model performance over time

## File Descriptions

- `01_data_exploration.ipynb`: Analyze dataset statistics and distributions
- `03_model_training.ipynb`: Full training pipeline (recommended to start here)
- `src/model.py`: CRNN architecture with two model options
- `src/dataset.py`: Custom dataset loader with augmentation
- `src/utils.py`: Helper functions (metrics, visualization, checkpoints)
- `data_processing/prepare_dataset.py`: Create train/val/test splits
- `configs/config.yaml`: Centralized configuration

## Performance Benchmarks

Expected results with default configuration:
- Character Error Rate (CER): 3-8%
- Training Time: 2-4 hours on Google Colab GPU
- Model Size: ~50MB
- Inference Speed: ~100 images/minute on GPU

## Contact & Support

For issues or questions:
1. Check the README.md file
2. Review configuration in config.yaml
3. Adjust hyperparameters based on your dataset
4. Ensure data format matches expected structure

## References

- CRNN: https://arxiv.org/abs/1507.05717
- CTC Loss: https://arxiv.org/abs/1311.2508
- Medical OCR: https://arxiv.org/abs/1907.09893
