"""
Comprehensive training script for OCR model
Can be run locally or on Google Colab
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = './checkpoints'
        self.log_interval = 100
        self.val_interval = 5


def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['images'].to(device)
        texts_tensor = batch['texts_tensor'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss (CTC)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) for CTC
        
        input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)
        loss = criterion(log_probs, texts_tensor, input_lengths, text_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % config.log_interval == 0:
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['images'].to(device)
            texts_tensor = batch['texts_tensor'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            log_probs = log_probs.permute(1, 0, 2)
            
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(log_probs, texts_tensor, input_lengths, text_lengths)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train OCR model')
    parser.add_argument('--data-dir', default='./data/processed', help='Data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.save_dir = args.checkpoint_dir
    
    logger.info(f"Configuration: Batch size={config.batch_size}, Epochs={config.num_epochs}")
    logger.info(f"Device: {config.device}")
    
    # Create checkpoint directory
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    # This is a template - actual implementation requires dataset loading
    logger.info("Training configuration ready!")
    logger.info("Please load your dataset and model before running training.")


if __name__ == '__main__':
    main()
