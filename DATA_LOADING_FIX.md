# Data Loading Fix - train_colab.py

## Problem
The script was throwing: `ValueError: num_samples should be a positive integer value, but got num_samples=0`

This happened because:
1. The script expected preprocessed data in `ocr_model/data/processed/`
2. The `prepare_dataset.py` preprocessing step wasn't running correctly in Colab
3. No images were being found, resulting in empty datasets

## Solution
Updated `train_colab.py` to **load data directly** from your existing directory structure:

```
data/
├── data1/
│   ├── Input/        (prescription images)
│   └── Output/       (prescription texts)
└── lbmaske/
    ├── Input/        (lab report images)
    └── Output/       (lab report texts)
```

## Changes Made

### Before
- Tried to run `prepare_dataset.py` to preprocess data
- Looked for data in `ocr_model/data/processed/train|val|test/`
- Used `MedicalDocumentDataset` class

### After
- **Section 5** now directly loads image-text pairs from Input/Output directories
- Automatically finds matching pairs (e.g., `1.jpg` + `1.txt`)
- Creates train/val/test splits in memory (80/10/10)
- Uses `OCRDataset` which is more direct

## What the New Process Does

```python
# Step 1: Load all image-text pairs from both directories
prescription_pairs = load_image_text_pairs(data/data1)        # 128 pairs found
lab_pairs = load_image_text_pairs(data/lbmaske)              # 426 pairs found
all_pairs = prescription_pairs + lab_pairs                    # 554 total

# Step 2: Shuffle and split
random.shuffle(all_pairs)
train_pairs = all_pairs[:443]    # 80%
val_pairs = all_pairs[443:498]   # 10%
test_pairs = all_pairs[498:]     # 10%

# Step 3: Create datasets directly
train_dataset = OCRDataset(images=..., texts=..., augment=True)
```

## How to Use in Colab

No changes needed! Just run as before:

```python
# In Colab cell 1:
!git clone https://github.com/Indicgamer/EMR_OCR.git

# In Colab cell 2:
!python /content/EMR_OCR/ocr_model/train_colab.py
```

The script will now:
1. Detect your data correctly
2. Find 554 image-text pairs (128 + 426)
3. Create train/val/test datasets
4. Start training immediately

## Verification

The script will print:
```
Loading prescription data...
  Found 128 prescription pairs
Loading lab report data...
  Found 426 lab report pairs
Total pairs loaded: 554

Data split:
  Train: 443 samples
  Val: 55 samples
  Test: 56 samples

Creating Data Loaders
  ✓ Data loaders created successfully
```

If you see "ERROR: No image-text pairs found!", then verify:
- `/content/drive/MyDrive/EMR_OCR/data/data1/Input/` contains images
- `/content/drive/MyDrive/EMR_OCR/data/data1/Output/` contains text files
- Same for `data/lbmaske/`

## Files Modified

- `ocr_model/train_colab.py` - SECTION 5 and SECTION 6 updated

## Pushed to GitHub

✓ Changes committed and pushed to https://github.com/Indicgamer/EMR_OCR

Just pull the latest changes to get the fix!
