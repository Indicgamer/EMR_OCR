import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
from pathlib import Path
import albumentations as A
from PIL import Image

class OCRDataset(Dataset):
    """
    Dataset class for OCR training on medical documents
    Loads image-text pairs and applies augmentation
    """
    
    def __init__(self, image_paths, texts, img_height=64, img_width=256, 
                 augment=False, transform=None):
        """
        Initialize OCR Dataset
        
        Args:
            image_paths: List of paths to images
            texts: List of corresponding text strings
            img_height: Target image height
            img_width: Target image width
            augment: Whether to apply data augmentation
            transform: Custom transform function
        """
        self.image_paths = image_paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.transform = transform
        
        # Build character set
        self.char_set = self._build_char_set()
        # IMPORTANT: Index 0 is reserved for CTC blank character
        # Character indices start from 1
        self.char2idx = {char: (idx + 1) for idx, char in enumerate(self.char_set)}
        self.idx2char = {(idx + 1): char for char, idx in self.char2idx.items()}
        # Add blank character mapping
        self.idx2char[0] = ''  # Blank character
        
        # Data augmentation pipeline
        self.augment_transforms = A.Compose([
            A.Rotate(limit=10, p=0.3),
            A.Affine(shear=10, p=0.2),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(p=0.1),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1))
        
        assert len(self.image_paths) == len(self.texts), \
            "Number of images must match number of texts"
    
    def _build_char_set(self):
        """Build character set from all texts"""
        chars = set()
        for text in self.texts:
            chars.update(text)
        
        # Add special characters
        chars.update([' ', '-', '/', '.', ',', '(', ')', '[', ']', '°', '+', '*'])
        
        # Sort for consistency
        return sorted(list(chars))
    
    def text_to_tensor(self, text):
        """Convert text to tensor indices"""
        indices = []
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            # Skip unknown characters - don't add anything
        
        # If text is empty after filtering, add a space
        if not indices:
            indices.append(self.char2idx[' '])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get single sample"""
        # Read image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Data augmentation
        if self.augment:
            augmented = self.augment_transforms(image=image)
            image = augmented['image']
        
        # Normalize to [0, 1] or [-1, 1]
        image = image.astype(np.float32) / 255.0
        
        # Custom transform
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Get text and convert to tensor
        text = self.texts[idx]
        text_tensor = self.text_to_tensor(text)
        
        return {
            'image': image,
            'text': text,
            'text_tensor': text_tensor,
            'image_path': image_path
        }


class MedicalDocumentDataset(Dataset):
    """
    Dataset for medical documents (prescriptions and lab reports)
    Handles loading image-text pairs from directory structure
    """
    
    def __init__(self, root_dir, img_height=64, img_width=256, augment=False):
        """
        Initialize Medical Document Dataset
        
        Args:
            root_dir: Root directory containing images and text files
            img_height: Target image height
            img_width: Target image width
            augment: Whether to apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        
        # Supported image formats
        self.img_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Load image-text pairs
        self.samples = self._load_samples()
        
        # Build character set
        self.char_set = self._build_char_set()
        # IMPORTANT: Index 0 is reserved for CTC blank character
        # Character indices start from 1
        self.char2idx = {char: (idx + 1) for idx, char in enumerate(self.char_set)}
        self.idx2char = {(idx + 1): char for char, idx in self.char2idx.items()}
        # Add blank character mapping
        self.idx2char[0] = ''  # Blank character
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Character set size: {len(self.char_set)}")
    
    def _load_samples(self):
        """Load all image-text pairs from directory"""
        samples = []
        
        # Find all images
        for img_file in self.root_dir.rglob('*'):
            if img_file.suffix.lower() in self.img_formats:
                # Look for corresponding text file
                txt_file = img_file.with_suffix('.txt')
                
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:  # Only add if text is not empty
                        samples.append({
                            'image_path': str(img_file),
                            'text': text
                        })
        
        return samples
    
    def _build_char_set(self):
        """Build character set from all texts"""
        chars = set()
        for sample in self.samples:
            chars.update(sample['text'])
        
        # Add special medical characters
        chars.update([' ', '-', '/', '.', ',', '(', ')', '[', ']', '°', '+', '*', ':', ';', '='])
        
        return sorted(list(chars))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get single sample"""
        sample = self.samples[idx]
        
        # Read image
        image_path = sample['image_path']
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Augmentation
        if self.augment:
            transform = A.Compose([
                A.Rotate(limit=10, p=0.3),
                A.Affine(shear=10, p=0.2),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
            ])
            image = transform(image=image)['image']
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Text to indices
        text = sample['text']
        text_indices = []
        for char in text:
            if char in self.char2idx:
                text_indices.append(self.char2idx[char])
        
        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        
        return {
            'image': image,
            'text': text,
            'text_tensor': text_tensor,
            'image_path': image_path
        }


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences
    """
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    texts_tensor = [item['text_tensor'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in texts_tensor], dtype=torch.long)
    max_text_length = max(text_lengths)
    
    padded_texts = []
    for t in texts_tensor:
        padded = torch.zeros(max_text_length, dtype=torch.long)
        padded[:len(t)] = t
        padded_texts.append(padded)
    
    padded_texts = torch.stack(padded_texts)
    
    return {
        'images': images,
        'texts': texts,
        'texts_tensor': padded_texts,
        'text_lengths': text_lengths,
        'image_paths': image_paths
    }


if __name__ == "__main__":
    # Test dataset loading
    root_dir = "../data/data1"
    
    try:
        dataset = MedicalDocumentDataset(root_dir, augment=False)
        print(f"Dataset size: {len(dataset)}")
        
        # Get first sample
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Text: {sample['text']}")
        print(f"Text tensor shape: {sample['text_tensor'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
