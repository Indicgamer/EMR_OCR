# !pip install -q transformers[torch] datasets evaluate jiwer accelerate
import os
import subprocess
from sklearn.model_selection import train_test_split

# 1. Setup Environment
# Clone the repository
if not os.path.exists('/content/EMR_OCR'):
    print("Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/Indicgamer/EMR_OCR.git', '/content/EMR_OCR'])

# Install dependencies from the requirements file
print("Installing dependencies...")
subprocess.run(['pip', 'install', '-r', '/content/EMR_OCR/requirements.txt'])

import sys
sys.path.append('/content/EMR_OCR')


from model import create_model
from dataset import get_image_text_pairs, MedicalDataset
from train import run_training

# 2. Load and Split Data
print("Loading data...")
image_paths, texts = get_image_text_pairs('/content/EMR_OCR')
train_imgs, val_imgs, train_txts, val_txts = train_test_split(image_paths, texts, test_size=0.15)

# 3. Initialize Model & Datasets
model, processor = create_model()
train_ds = MedicalDataset(train_imgs, train_txts, processor)
val_ds = MedicalDataset(val_imgs, val_txts, processor)

# 4. Start Training
print(f"Starting training on {len(train_imgs)} samples...")
trainer = run_training(
    model=model, 
    processor=processor, 
    train_dataset=train_ds, 
    val_dataset=val_ds, 
    output_dir="./trocr_emr_results"
)

# 5. Save the final model
trainer.save_model("./final_medical_ocr_model")
print("Training Complete! Model saved to ./final_medical_ocr_model")