# OCR Model Development - File Reference

## Complete File Listing with Descriptions

### Root Directory Files

```
ocr_model/
├── requirements.txt              # Python package dependencies
├── README.md                     # Comprehensive documentation
├── SETUP.md                      # Installation & setup guide
├── QUICKSTART.md                 # Quick start for Colab
└── PROJECT_SUMMARY.md            # This file - project overview
```

## Source Code (src/)

### model.py
**Purpose**: Model architectures for OCR

**Key Classes**:
- `CRNN`: Full CRNN model with ResNet-34 backbone
  - Input: (batch, 3, 64, 256)
  - Output: (batch, sequence_length, num_classes)
  - Parameters: ~25M
  
- `OCRModel`: Lightweight MobileNetV2-based model
  - Faster inference
  - Smaller model size
  - Parameters: ~10M

**Key Functions**:
- `create_model()`: Factory function to instantiate models

**Usage**:
```python
from src.model import create_model
model = create_model("CRNN", num_classes=95)
```

### dataset.py
**Purpose**: Dataset loading and preprocessing

**Key Classes**:
- `OCRDataset`: Generic image-text pair dataset
  - Handles image loading and resizing
  - Text encoding/decoding
  - Data augmentation
  
- `MedicalDocumentDataset`: EMR-specific dataset
  - Loads from directory structure
  - Automatic character set building
  - Handles medical documents

**Key Functions**:
- `collate_fn()`: Custom batch collation for variable-length sequences

**Usage**:
```python
from src.dataset import MedicalDocumentDataset
dataset = MedicalDocumentDataset('./data/processed/train')
```

### utils.py
**Purpose**: Utility and helper functions

**Key Classes**:
- `CharacterSet`: Manages character-to-index mapping
  - Encoding: text → indices
  - Decoding: indices → text

**Key Functions**:
- `load_image()`: Load and preprocess images
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load saved model
- `calculate_cer()`: Character Error Rate
- `calculate_wer()`: Word Error Rate
- `get_device()`: Get available device (GPU/CPU)
- `print_model_info()`: Display model architecture
- `count_parameters()`: Count trainable params
- `edit_distance()`: Levenshtein distance calculation

**Usage**:
```python
from src.utils import get_device, calculate_cer
device = get_device()
cer = calculate_cer(predictions, ground_truths)
```

### train.py
**Purpose**: Training script template

**Key Classes**:
- `TrainingConfig`: Configuration holder

**Key Functions**:
- `train_epoch()`: Single epoch training
- `validate()`: Validation loop
- `main()`: Entry point with argument parsing

**Usage**:
```bash
python src/train.py --epochs 50 --batch-size 32 --lr 1e-4
```

### inference.py
**Purpose**: Inference engine for predictions

**Key Classes**:
- `OCRInference`: Inference pipeline
  - Model loading
  - Image preprocessing
  - Prediction generation

**Key Functions**:
- `preprocess_image()`: Prepare image for inference
- `predict()`: Generate predictions

**Usage**:
```python
from src.inference import OCRInference
ocr = OCRInference('model.pth', model_class, charset)
text = ocr.predict('image.png')
```

### __init__.py
**Purpose**: Package initialization

**Exports**: All main classes and functions for easy import

**Usage**:
```python
from src import CRNN, OCRDataset, calculate_cer
```

## Data Processing (data_processing/)

### prepare_dataset.py
**Purpose**: Prepare dataset for training

**Functionality**:
1. Scans prescription and lab report directories
2. Finds image-text pairs
3. Creates train/val/test splits
4. Copies files to structured directories
5. Generates manifest files

**Usage**:
```bash
python data_processing/prepare_dataset.py \
  --prescriptions-dir ../data/data1 \
  --lab-reports-dir ../data/lbmaske \
  --output-dir ./data/processed \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

**Output Structure**:
```
data/processed/
├── train/
│   ├── images/
│   ├── texts/
│   └── manifest.json
├── val/
│   ├── images/
│   ├── texts/
│   └── manifest.json
└── test/
    ├── images/
    ├── texts/
    └── manifest.json
```

### extract_images.py
**Purpose**: Extract images from PDF files (optional)

**Functionality**:
- Converts PDF pages to images
- Batch processing
- Adjustable DPI and format

**Usage**:
```bash
python data_processing/extract_images.py \
  --pdf-dir ./pdfs \
  --output-dir ./images \
  --dpi 300 \
  --format png
```

## Configuration (configs/)

### config.yaml
**Purpose**: Centralized model and training configuration

**Sections**:
1. **dataset**: Data paths and preprocessing
   - Image dimensions: 64×256
   - Data splits: 80/10/10
   - Augmentation settings

2. **model**: Model architecture settings
   - Type: CRNN
   - Hidden size: 256
   - Num classes: 95

3. **training**: Training hyperparameters
   - Batch size: 32
   - Epochs: 50
   - Learning rate: 1e-4

4. **validation**: Evaluation settings
   - Metrics: CER, WER, Accuracy
   - Eval interval: every 5 epochs

5. **augmentation**: Data augmentation parameters
   - Rotation, shear, brightness, contrast, blur, noise

6. **hardware**: Device and logging
   - Device: cuda/cpu
   - Num workers: 4

7. **medical**: Medical-specific settings
   - Special characters for medical documents
   - Document types

**Usage**:
```python
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

## Notebooks (notebooks/)

### 01_data_exploration.ipynb
**Purpose**: Analyze and explore dataset

**Sections**:
1. Setup and installation
2. Data loading and exploration
3. Dataset statistics
4. Text length distribution
5. Character set analysis

**Outputs**:
- Dataset size statistics
- Text length histograms
- Character set display

**Time**: 5-10 minutes

### 03_model_training.ipynb
**Purpose**: Complete training pipeline

**Sections**:
1. Setup Colab environment
2. Data preparation
3. Model architecture setup
4. Training configuration
5. Training loop (main)
6. Training evaluation
7. Model testing and inference

**Outputs**:
- Trained model: checkpoints/best_model.pth
- Training history: checkpoints/training_history.json
- Performance metrics: CER, loss curves

**Time**: 2-4 hours on Google Colab GPU

### 02_data_preprocessing.ipynb (Optional)
**Purpose**: Detailed data preprocessing steps

**Includes**:
- Image augmentation examples
- Text preprocessing techniques
- Batch creation
- Visualization

### 04_evaluation_inference.ipynb (Optional)
**Purpose**: Detailed evaluation and inference

**Includes**:
- Model evaluation metrics
- Error analysis
- Sample predictions
- Confusion matrices

## Checkpoint Directory (checkpoints/)

### best_model.pth
**Format**: PyTorch checkpoint dictionary

**Contents**:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'metrics': dict
}
```

**Size**: ~50MB

**Usage**:
```python
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### training_history.json
**Format**: JSON file

**Contents**:
```json
{
    "train_loss": [float, ...],
    "val_loss": [float, ...],
    "epoch": [int, ...]
}
```

**Usage**:
```python
import json
with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)
```

## Data Directory Structure

### data/data1/ (Prescriptions)
```
data/data1/
├── Input/                    # Prescription images (optional)
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── Output/                   # Prescription texts (required)
    ├── 1.txt
    ├── 2.txt
    └── ... (up to 130.txt)
```

### data/lbmaske/ (Lab Reports)
```
data/lbmaske/
├── Input/                    # Lab report images (optional)
└── Output/                   # Lab report texts (required)
    ├── AHD-0425-PA-0007561_JITENDRA_TRIVEDI_DS_28-04-2025_1019-21_AM.pdf_page_9.txt
    ├── AHD-0425-PA-0007719_E-REPORTS_250427_2032@E.pdf_page_4.txt
    └── ... (~500+ files)
```

### data/processed/ (After prepare_dataset.py)
```
data/processed/
├── train/
│   ├── images/               # Training images
│   ├── texts/                # Training texts
│   └── manifest.json         # File mapping
├── val/
│   ├── images/
│   ├── texts/
│   └── manifest.json
└── test/
    ├── images/
    ├── texts/
    └── manifest.json
```

## Documentation Files

### README.md
**Content**:
- Project overview
- Architecture explanation
- Setup instructions
- Workflow description
- Model approaches (3 options)
- Training tips
- Expected outputs
- References

**Length**: Comprehensive (1000+ lines)

### QUICKSTART.md
**Content**:
- Quick start guide for Colab
- 5-step setup process
- Configuration parameters
- Common issues & solutions
- Model architecture diagram
- Training tips
- Next steps

**Length**: Medium (~300 lines)
**Best for**: Users who want to start immediately

### SETUP.md
**Content**:
- Detailed installation instructions
- System requirements
- Step-by-step setup
- Configuration guide
- Directory structure
- Troubleshooting
- Data format requirements
- Performance monitoring

**Length**: Detailed (~400 lines)
**Best for**: First-time setup

### PROJECT_SUMMARY.md
**Content**:
- Complete project overview
- What's included
- Model architecture details
- Training pipeline
- Usage examples
- Performance expectations
- File organization
- Technology stack
- Next steps

**Length**: Comprehensive (~500 lines)
**Best for**: Understanding project architecture

## Usage Flow

### 1. Initial Setup
- Read: SETUP.md
- Run: `prepare_dataset.py`

### 2. Data Exploration
- Open: `01_data_exploration.ipynb`
- Understand data characteristics

### 3. Training
- Open: `03_model_training.ipynb`
- Execute cells sequentially
- Monitor training progress

### 4. Evaluation
- Check: `checkpoints/training_history.json`
- Review: Character Error Rate
- Analyze: Sample predictions

### 5. Deployment
- Load: `checkpoints/best_model.pth`
- Use: `src/inference.py`
- Integrate with EMR system

## Quick Reference

### Key Commands

```bash
# Setup
pip install -r requirements.txt

# Prepare data
python data_processing/prepare_dataset.py

# Train
python src/train.py --epochs 50

# In Python
from src import CRNN, calculate_cer, get_device
```

### Key Classes

```python
CRNN                          # Main model
MedicalDocumentDataset        # Data loader
OCRInference                  # Inference engine
CharacterSet                  # Encoding/decoding
TrainingConfig                # Training settings
```

### Key Functions

```python
create_model()               # Create model instance
calculate_cer()              # Character Error Rate
get_device()                 # Get GPU/CPU
load_checkpoint()            # Load saved model
save_checkpoint()            # Save model
```

---

**For specific questions**:
- Installation: See SETUP.md
- Quick start: See QUICKSTART.md
- Project overview: See PROJECT_SUMMARY.md
- Full documentation: See README.md
