# EMR OCR Model Development

Custom OCR model for recognizing text from medical documents (prescriptions and lab reports) for the Electronic Medical Record (EMR) system.

## Project Structure

```
ocr_model/
├── notebooks/              # Google Colab notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_inference.ipynb
├── src/                    # Core model code
│   ├── __init__.py
│   ├── model.py           # Model architecture
│   ├── dataset.py         # Dataset classes
│   └── utils.py           # Utility functions
├── data_processing/       # Data preparation scripts
│   ├── extract_images.py  # Extract images from PDFs
│   └── prepare_dataset.py # Create train/val/test splits
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   └── model_config.yaml  # Model-specific config
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Data Structure

- **Prescriptions**: `data/data1/` - 130 text files with corresponding images
- **Lab Reports**: `data/lbmaske/` - Extracted text from PDF pages (image text mapping)

## Setup Instructions

### Local Setup
```bash
# Clone the repository
cd ocr_model

# Install dependencies
pip install -r requirements.txt

# Set up data directories
python data_processing/prepare_dataset.py
```

### Google Colab Setup
1. Upload the project to Google Drive
2. Mount Drive in Colab notebook
3. Install requirements in Colab:
```python
!pip install -r /content/drive/MyDrive/new_EMR/ocr_model/requirements.txt
```

## Workflow

### 1. Data Exploration
- Analyze available prescription images and lab reports
- Understand text distribution and document types
- Check data quality and OCR challenges

### 2. Data Preprocessing
- Extract images from PDFs (if needed)
- Create image-text pairs
- Data augmentation
- Create train/validation/test splits

### 3. Model Training
- Choose OCR architecture (CRNN, EfficientNet, or PaddleOCR-based)
- Train on prescription and lab report datasets
- Handle custom medical terminology

### 4. Evaluation & Inference
- Evaluate Character Error Rate (CER) and Word Error Rate (WER)
- Test on prescription images
- Test on lab report images
- Generate predictions

## Model Approaches

### Option 1: PaddleOCR (Recommended for beginners)
- Pre-trained multilingual OCR
- Fine-tune on medical documents
- Good out-of-box performance

### Option 2: CRNN + CTC Loss
- Custom architecture for character recognition
- Sequence-to-sequence with attention
- Better for domain-specific training

### Option 3: Transformer-based OCR
- Modern architecture with attention mechanisms
- Best accuracy but requires more data
- Suitable for medical document OCR

## Training Tips

1. **Data Augmentation**: Use rotation, shear, brightness adjustments for robustness
2. **Learning Rate**: Start with 1e-4, reduce on plateau
3. **Batch Size**: Use 32-64 depending on GPU memory
4. **Early Stopping**: Monitor validation loss
5. **Medical Terminology**: Collect OCR errors and retrain

## Expected Outputs

- Trained model weights (`.pth` or `.pb` file)
- Evaluation metrics (CER, WER, accuracy)
- Predictions on test set
- Error analysis report

## References

- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [CRNN: An End-to-End Trainable Neural Network](https://arxiv.org/abs/1507.05717)
- [Transformer for OCR](https://arxiv.org/abs/2103.06450)
