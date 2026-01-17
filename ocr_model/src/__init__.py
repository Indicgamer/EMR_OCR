"""
Initialize OCR model package
"""

__version__ = "0.1.0"
__author__ = "EMR OCR Team"

from .model import CRNN, OCRModel, create_model
from .dataset import OCRDataset, MedicalDocumentDataset, collate_fn
from .utils import (
    CharacterSet, load_image, save_checkpoint, load_checkpoint,
    calculate_cer, calculate_wer, get_device, count_parameters
)

__all__ = [
    'CRNN',
    'OCRModel',
    'create_model',
    'OCRDataset',
    'MedicalDocumentDataset',
    'collate_fn',
    'CharacterSet',
    'load_image',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_cer',
    'calculate_wer',
    'get_device',
    'count_parameters'
]
