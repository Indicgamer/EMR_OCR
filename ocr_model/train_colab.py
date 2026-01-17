"""
OCR Model Training Script for Google Colab
Run this script directly in Colab to train your OCR model
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# ============================================================================
# SECTION 1: Setup Google Colab Environment
# ============================================================================
print("=" * 80)
print("SECTION 1: Setup Google Colab Environment")
print("=" * 80)

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# SECTION 2: Install Dependencies
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Installing Dependencies")
print("=" * 80)

packages = ['albumentations', 'tensorboard', 'pyyaml']
for pkg in packages:
    subprocess.check_call(['pip', 'install', '-q', pkg])

print("Dependencies installed successfully!")

# ============================================================================
# SECTION 3: Mount Google Drive & Set Paths
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Mount Google Drive & Set Paths")
print("=" * 80)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive/EMR_OCR'
    print(f"Google Drive mounted at: {base_path}")
except:
    # If not in Colab, use current directory
    base_path = '/content/EMR_OCR' if os.path.exists('/content/EMR_OCR') else '.'
    print(f"Using local path: {base_path}")

# Add project to path
sys.path.insert(0, f'{base_path}/ocr_model/src')
sys.path.insert(0, f'{base_path}/ocr_model')

# ============================================================================
# SECTION 4: Import Libraries
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Import Libraries")
print("=" * 80)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

try:
    from model import create_model
    from dataset import OCRDataset, collate_fn
    from utils import get_device, save_checkpoint, calculate_cer, print_model_info
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all files are in correct location")
    sys.exit(1)

# ============================================================================
# SECTION 5: Load Raw Data and Create Splits
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Load Raw Data and Create Splits")
print("=" * 80)

import random
from pathlib import Path

def load_image_text_pairs(data_dir):
    """Load image-text pairs directly from Input/Output structure"""
    pairs = []
    input_dir = Path(data_dir) / 'Input'
    output_dir = Path(data_dir) / 'Output'
    
    if not input_dir.exists() or not output_dir.exists():
        print(f"Warning: {data_dir} structure not found")
        return []
    
    # Find all image files
    for img_file in input_dir.glob('*'):
        if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            # Get corresponding text file
            txt_file = output_dir / (img_file.stem + '.txt')
            
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                    
                    # Only include non-empty texts
                    if text and len(text) > 0:
                        pairs.append({
                            'image': str(img_file),
                            'text': text
                        })
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
    
    return pairs

# Load data from both directories
print("Loading prescription data...")
prescription_pairs = load_image_text_pairs(f'{base_path}/data/data1')
print(f"  Found {len(prescription_pairs)} prescription pairs")

print("Loading lab report data...")
lab_pairs = load_image_text_pairs(f'{base_path}/data/lbmaske')
print(f"  Found {len(lab_pairs)} lab report pairs")

# Combine and shuffle all data
all_pairs = prescription_pairs + lab_pairs
print(f"Total pairs loaded: {len(all_pairs)}")

if len(all_pairs) == 0:
    print("ERROR: No image-text pairs found!")
    print(f"  Prescription Input: {Path(f'{base_path}/data/data1/Input').exists()}")
    print(f"  Prescription Output: {Path(f'{base_path}/data/data1/Output').exists()}")
    print(f"  Lab Input: {Path(f'{base_path}/data/lbmaske/Input').exists()}")
    print(f"  Lab Output: {Path(f'{base_path}/data/lbmaske/Output').exists()}")
    sys.exit(1)

random.seed(42)
random.shuffle(all_pairs)

# Create train/val/test splits (80/10/10)
train_size = int(0.8 * len(all_pairs))
val_size = int(0.1 * len(all_pairs))

train_pairs = all_pairs[:train_size]
val_pairs = all_pairs[train_size:train_size + val_size]
test_pairs = all_pairs[train_size + val_size:]

print(f"\nData split:")
print(f"  Train: {len(train_pairs)} samples")
print(f"  Val: {len(val_pairs)} samples")
print(f"  Test: {len(test_pairs)} samples")

# ============================================================================
# SECTION 6: Load Datasets
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Load Datasets")
print("=" * 80)

# Create datasets directly from pairs
train_dataset = OCRDataset(
    image_paths=[p['image'] for p in train_pairs],
    texts=[p['text'] for p in train_pairs],
    img_height=64,
    img_width=256,
    augment=True
)

val_dataset = OCRDataset(
    image_paths=[p['image'] for p in val_pairs],
    texts=[p['text'] for p in val_pairs],
    img_height=64,
    img_width=256,
    augment=False
)

test_dataset = OCRDataset(
    image_paths=[p['image'] for p in test_pairs],
    texts=[p['text'] for p in test_pairs],
    img_height=64,
    img_width=256,
    augment=False
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Character set size: {len(train_dataset.char_set)}")

# ============================================================================
# SECTION 7: Create Data Loaders
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Create Data Loaders")
print("=" * 80)

batch_size = 32
num_workers = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=True
)

print(f"Data loaders created successfully!")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# SECTION 8: Model Architecture Setup
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: Model Architecture Setup")
print("=" * 80)

device = get_device()

# Create model
model = create_model(
    model_type="CRNN",
    num_classes=len(train_dataset.char_set),
    pretrained=True
)

model = model.to(device)

# Print model info
print_model_info(model)

# ============================================================================
# SECTION 9: Training Configuration
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: Training Configuration")
print("=" * 80)

num_epochs = 30
learning_rate = 1e-4
weight_decay = 1e-5

# Loss function (CTC for sequence-to-sequence)
criterion = nn.CTCLoss(reduction='mean', zero_infinity=True)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

print("Training configuration ready!")
print(f"Optimizer: Adam")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {num_epochs}")
print(f"Loss function: CTCLoss")
print(f"Device: {device}")

# ============================================================================
# SECTION 10: Create Checkpoint Directory
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: Setup Checkpoints")
print("=" * 80)

checkpoint_dir = f'{base_path}/ocr_model/checkpoints'
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'epoch': []
}

best_val_loss = float('inf')
patience = 5
patience_counter = 0

print(f"Checkpoint directory: {checkpoint_dir}")

# ============================================================================
# SECTION 11: Training Loop
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: Training Loop")
print("=" * 80)

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(train_bar):
        images = batch['images'].to(device)
        texts_tensor = batch['texts_tensor'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Prepare for CTC loss
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) required by CTC
        
        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(1),
            dtype=torch.long,
            device=device
        )
        
        # Calculate loss
        loss = criterion(log_probs, texts_tensor, input_lengths, text_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            train_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'lr': scheduler.get_last_lr()[0]
            })
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    val_bar = tqdm(val_loader, desc='Validating')
    with torch.no_grad():
        for batch in val_bar:
            images = batch['images'].to(device)
            texts_tensor = batch['texts_tensor'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            log_probs = log_probs.permute(1, 0, 2)
            
            input_lengths = torch.full(
                (images.size(0),),
                outputs.size(1),
                dtype=torch.long,
                device=device
            )
            
            loss = criterion(log_probs, texts_tensor, input_lengths, text_lengths)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Update scheduler
    scheduler.step()
    
    # Record history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['epoch'].append(epoch + 1)
    
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        save_checkpoint(
            model, optimizer, epoch,
            {'train_loss': avg_train_loss, 'val_loss': avg_val_loss},
            f'{checkpoint_dir}/best_model.pth'
        )
        print(f"✓ Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\nTraining completed!")

# ============================================================================
# SECTION 12: Evaluate Training
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12: Evaluate Training")
print("=" * 80)

# Plot training history
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
ax.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training History - OCR Model')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{checkpoint_dir}/training_history.png')
plt.show()

print(f"Best validation loss: {min(history['val_loss']):.4f}")

# Save training history
history_path = f'{checkpoint_dir}/training_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"Training history saved to: {history_path}")

# ============================================================================
# SECTION 13: Model Testing and Inference
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 13: Model Testing and Inference")
print("=" * 80)

# Load best model
best_checkpoint = torch.load(f'{checkpoint_dir}/best_model.pth', map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Best model loaded for inference")

# Get predictions on test set
print("\nGenerating predictions...")
predictions = []
ground_truths = []

def decode_ctc_output(predicted_indices, idx2char, blank_idx=0):
    """Decode CTC output by removing blanks and consecutive duplicates"""
    decoded = []
    prev_idx = None
    
    for idx in predicted_indices:
        idx_val = idx.item()
        # Skip blank characters (index 0)
        if idx_val != blank_idx:
            # Skip consecutive duplicates
            if idx_val != prev_idx:
                char = idx2char.get(idx_val, '')
                if char:  # Only add non-empty characters
                    decoded.append(char)
            prev_idx = idx_val
        else:
            prev_idx = None
    
    return ''.join(decoded)

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        images = batch['images'].to(device)
        outputs = model(images)
        
        # Decode predictions using CTC rules
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        predicted_indices = torch.argmax(log_probs, dim=2)
        
        for idx_seq in predicted_indices:
            text = decode_ctc_output(idx_seq, train_dataset.idx2char, blank_idx=0)
            predictions.append(text.strip())
        
        # Get ground truth
        for text in batch['texts']:
            ground_truths.append(text)

print(f"Generated {len(predictions)} predictions")
print(f"Ground truth samples: {len(ground_truths)}")

# ============================================================================
# SECTION 14: Calculate Metrics
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 14: Calculate Metrics")
print("=" * 80)

# Calculate CER (Character Error Rate)
cer = calculate_cer(predictions, ground_truths)
print(f"\nCharacter Error Rate (CER): {cer*100:.2f}%")

# Show sample predictions
print("\n" + "="*80)
print("Sample Predictions (First 5)")
print("="*80)
for i in range(min(5, len(predictions))):
    print(f"\nSample {i+1}:")
    print(f"Ground truth: {ground_truths[i][:100]}")
    print(f"Prediction : {predictions[i][:100]}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n✅ Model saved to: {checkpoint_dir}/best_model.pth")
print(f"✅ Training history saved to: {checkpoint_dir}/training_history.json")
print(f"✅ Final CER: {cer*100:.2f}%")
print(f"✅ Best validation loss: {min(history['val_loss']):.4f}")
print("\nYour OCR model is ready for deployment! 🚀")
