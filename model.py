import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class MedicalDataset(Dataset):
    def __init__(self, image_paths, texts, processor, max_target_length=64):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess image (resizes to 384x384)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize text
        labels = self.processor.tokenizer(
            self.texts[idx], 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace padding with -100 so it is ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

def get_image_text_pairs(base_path):
    """Parses the data1 and lbmaske directories for Input/Output pairs"""
    image_paths = []
    texts = []
    data_dirs = [Path(base_path) / 'data/data1', Path(base_path) / 'data/lbmaske']
    
    for d in data_dirs:
        input_dir = d / 'Input'
        output_dir = d / 'Output'
        if not input_dir.exists(): continue
        
        for img_file in input_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                txt_file = output_dir / (img_file.stem + '.txt')
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                    if text:
                        image_paths.append(str(img_file))
                        texts.append(text)
    return image_paths, texts