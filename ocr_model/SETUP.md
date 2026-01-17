# OCR Model Development - Setup & Installation Guide

## System Requirements

### Local Machine (Optional)
- Python 3.8+
- GPU with CUDA support (optional, for faster training)
- 8GB RAM minimum
- 10GB disk space for datasets

### Google Colab (Recommended)
- Free Google account
- Google Drive access
- GPU runtime (free tier available)

## Installation Steps

### Option 1: Google Colab (RECOMMENDED)

#### 1. Prepare Files
- Upload `new_EMR` folder to Google Drive
- Keep structure intact

#### 2. Open Notebook
- Go to [colab.research.google.com](https://colab.research.google.com)
- Open `new_EMR/ocr_model/notebooks/03_model_training.ipynb`
- Or upload via File → Open notebook → GitHub tab

#### 3. Enable GPU
- Runtime → Change runtime type
- Hardware accelerator → GPU
- Click Save

#### 4. Execute Notebook
- Click "Run all" or execute cells sequentially
- Monitor output for any errors
- Training will start automatically

### Option 2: Local Machine

#### 1. Clone/Download Project
```bash
cd path/to/new_EMR/ocr_model
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Prepare Dataset
```bash
python data_processing/prepare_dataset.py \
  --prescriptions-dir ../data/data1 \
  --lab-reports-dir ../data/lbmaske \
  --output-dir ./data/processed
```

#### 5. Run Training
```bash
python src/train.py \
  --data-dir ./data/processed \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-4
```

## Configuration

### Edit Model Configuration
File: `configs/config.yaml`

Key parameters:
```yaml
# Dataset
dataset:
  resize_height: 64           # Image height
  resize_width: 256          # Image width
  train_split: 0.8           # 80% training

# Model
model:
  type: "CRNN"               # Architecture type
  hidden_size: 256           # LSTM hidden units
  num_classes: 95            # Character classes

# Training
training:
  batch_size: 32             # Batch size
  num_epochs: 50             # Max epochs
  learning_rate: 1e-4        # Initial LR
  optimizer: "Adam"          # Optimizer type
```

## Directory Structure After Setup

```
new_EMR/
├── data/
│   ├── data1/
│   │   ├── Input/           # Prescription images
│   │   └── Output/          # Prescription texts
│   └── lbmaske/
│       ├── Input/           # Lab report images
│       └── Output/          # Lab report texts
│
└── ocr_model/
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   └── 03_model_training.ipynb
    ├── src/
    │   ├── model.py
    │   ├── dataset.py
    │   ├── utils.py
    │   ├── train.py
    │   └── inference.py
    ├── data_processing/
    │   ├── prepare_dataset.py
    │   └── extract_images.py
    ├── configs/
    │   └── config.yaml
    ├── checkpoints/         # ← Models save here
    ├── requirements.txt
    ├── README.md
    └── QUICKSTART.md
```

## Dependency Information

### Core Libraries
- `torch>=2.0.0`: Deep learning framework
- `torchvision>=0.15.0`: Computer vision utilities
- `opencv-python>=4.8.0`: Image processing
- `Pillow>=10.0.0`: Image library
- `albumentations>=1.3.0`: Data augmentation

### Data Processing
- `numpy>=1.24.0`: Numerical computing
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: ML utilities

### Utilities
- `matplotlib>=3.7.0`: Plotting
- `tqdm>=4.66.0`: Progress bars
- `pyyaml>=6.0`: Configuration files

### Optional (For PDF Processing)
- `pdf2image`: Convert PDF to images
- `pytesseract`: Tesseract OCR interface

## Troubleshooting

### Issue: Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Dataset Not Found
```bash
# Check paths are correct
ls data/data1/Output/
ls data/lbmaske/Output/
```

### Issue: CUDA Not Available
```python
# Force CPU (in notebook)
import torch
device = torch.device('cpu')
```

### Issue: Out of Memory
```python
# Reduce batch size in training
batch_size = 16  # or lower
```

## GPU Optimization Tips

### For Google Colab
- Batch size: 32 (for 15GB GPU memory)
- Use GPU notebook for 12+ hour training
- Monitor memory in Runtime → Manage sessions

### For Local GPU
- Check CUDA installation:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Update GPU drivers if needed

## Data Format Requirements

### Text Files
- Location: `data/data1/Output/` and `data/lbmaske/Output/`
- Format: UTF-8 plain text
- Filename: `{name}.txt`
- Content: Extracted text from documents

### Image Files (Optional)
- Location: `data/data1/Input/` and `data/lbmaske/Input/`
- Format: PNG, JPG, BMP, TIFF
- Resolution: Any (will be resized to 64×256)
- Color: RGB or Grayscale (will be converted)

## Next Steps

1. **Choose Execution Environment**:
   - Google Colab: Start with `03_model_training.ipynb`
   - Local: Run `python src/train.py`

2. **Explore Dataset**:
   - Open `01_data_exploration.ipynb` first
   - Understand data characteristics
   - Check for data issues

3. **Prepare Data**:
   - Run `prepare_dataset.py`
   - Creates train/val/test splits
   - Validates file structure

4. **Train Model**:
   - Execute training notebook
   - Monitor loss curves
   - Wait for convergence

5. **Evaluate Results**:
   - Check Character Error Rate
   - Review sample predictions
   - Save best model

## Performance Monitoring

### During Training
- Watch GPU memory usage (shouldn't exceed 80%)
- Monitor loss convergence
- Check validation accuracy

### After Training
- Review loss curves
- Calculate Character Error Rate (CER)
- Analyze error patterns
- Compare predictions with ground truth

## Model Export

### Save for Deployment
```python
# Already saved to: checkpoints/best_model.pth
```

### Load for Inference
```python
from src.inference import OCRInference
ocr = OCRInference('checkpoints/best_model.pth', model_class, charset)
result = ocr.predict('image.png')
```

## Getting Help

### Documentation
1. README.md - Detailed documentation
2. QUICKSTART.md - Quick reference
3. Code comments in src/*.py

### Common Issues
- See QUICKSTART.md "Common Issues & Solutions"
- Check requirements match your Python version
- Ensure all data paths are correct

### Debugging
```python
# In notebook, enable debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next: Start Training!

Once setup is complete:
1. Go to `QUICKSTART.md`
2. Follow "Quick Start - Google Colab" section
3. Run the training notebook step by step
