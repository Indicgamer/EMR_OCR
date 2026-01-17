import os
import torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import Dataset, load_metric

# ==========================================
# 1. LOAD DATA FROM YOUR STRUCTURE
# ==========================================
def load_medical_data(base_path):
    image_paths = []
    texts = []
    
    # Paths from your previous structure
    data_dirs = [
        Path(base_path) / 'data/data1', 
        Path(base_path) / 'data/lbmaske'
    ]
    
    for d in data_dirs:
        input_dir = d / 'Input'
        output_dir = d / 'Output'
        
        if not input_dir.exists(): continue
        
        for img_file in input_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                txt_file = output_dir / (img_file.stem + '.txt')
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        image_paths.append(str(img_file))
                        texts.append(text)
    
    return image_paths, texts

# Load your data (Update base_path if needed)
base_path = '/content/EMR_OCR' 
image_paths, texts = load_medical_data(base_path)

# Split into Train and Val
train_imgs, val_imgs, train_texts, val_texts = train_test_split(
    image_paths, texts, test_size=0.1, random_state=42
)

# ==========================================
# 2. INITIALIZE PROCESSOR & MODEL
# ==========================================
# 'microsoft/trocr-base-handwritten' is great for both print and handwriting
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Set model configurations
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64 # Adjust based on your longest text
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# ==========================================
# 3. PREPARE DATASET FOR HUGGINGFACE
# ==========================================
def preprocess(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    
    labels = processor.tokenizer(
        examples['text'], 
        padding="max_length", 
        max_length=model.config.max_length
    ).input_ids
    
    # Important: Mask the padding for the loss function
    labels = [[(l if l != processor.tokenizer.pad_token_id else -100) for l in label] for label in labels]
    
    return {"pixel_values": pixel_values, "labels": labels}

train_ds = Dataset.from_dict({"image_path": train_imgs, "text": train_texts})
val_ds = Dataset.from_dict({"image_path": val_imgs, "text": val_texts})

train_ds = train_ds.with_transform(preprocess)
val_ds = val_ds.with_transform(preprocess)

# ==========================================
# 4. METRICS (CER)
# ==========================================
cer_metric = load_metric("jiwer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# ==========================================
# 5. TRAINER
# ==========================================
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=4, # Keep low for Colab T4
    per_device_eval_batch_size=4,
    fp16=True, # Use GPU acceleration
    output_dir="./trocr_medical",
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    num_train_epochs=20, # TrOCR converges fast
    weight_decay=0.01,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
)

trainer.train()