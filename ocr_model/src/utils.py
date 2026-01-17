import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from PIL import Image
import io

class CharacterSet:
    """Handle character set management"""
    
    def __init__(self, characters: str = None):
        """
        Initialize character set
        
        Args:
            characters: String containing all valid characters
        """
        if characters is None:
            # Default character set: alphanumeric + medical special chars
            characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
                        " .,()[]/-+*°%@™®©:;=?\n"
        
        self.characters = sorted(list(set(characters)))
        self.char2idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to character indices"""
        return [self.char2idx.get(char, 0) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert character indices to text"""
        return ''.join([self.idx2char.get(idx, '') for idx in indices])
    
    def __len__(self):
        return len(self.characters)


def load_image(image_path: Union[str, Path], height: int = 64, width: int = 256) -> np.ndarray:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        height: Target height
        width: Target width
        
    Returns:
        Normalized image array [0, 1]
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32) / 255.0
    
    return image


def save_checkpoint(model, optimizer, epoch, metrics, save_path: Union[str, Path]):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, checkpoint_path: Union[str, Path], device: str = 'cpu'):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Loaded model and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Character Error Rate (CER)
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Character Error Rate (0-1)
    """
    total_errors = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        total_chars += len(target)
        # Simple edit distance (Levenshtein distance)
        errors = edit_distance(pred, target)
        total_errors += errors
    
    return total_errors / max(total_chars, 1)


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Word Error Rate (WER)
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Word Error Rate (0-1)
    """
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        total_words += len(target_words)
        errors = edit_distance_words(pred_words, target_words)
        total_errors += errors
    
    return total_errors / max(total_words, 1)


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def edit_distance_words(words1: List[str], words2: List[str]) -> int:
    """
    Calculate edit distance at word level
    """
    if len(words1) < len(words2):
        return edit_distance_words(words2, words1)
    
    if len(words2) == 0:
        return len(words1)
    
    previous_row = range(len(words2) + 1)
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def get_device():
    """Get available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def plot_training_history(history: Dict[str, List[float]], save_path: Union[str, Path] = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def visualize_predictions(images: np.ndarray, predictions: List[str], 
                         targets: List[str], save_path: Union[str, Path] = None):
    """
    Visualize OCR predictions
    
    Args:
        images: Batch of images
        predictions: Predicted texts
        targets: Ground truth texts
        save_path: Optional path to save figure
    """
    num_samples = min(len(predictions), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        image = images[idx]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        image = np.clip(image, 0, 1)
        
        axes[idx].imshow(image)
        axes[idx].set_title(
            f"Pred: {predictions[idx][:20]}\nTrue: {targets[idx][:20]}",
            fontsize=8
        )
        axes[idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(num_samples, 8):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print model architecture and parameter count"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model:\n{model}")
