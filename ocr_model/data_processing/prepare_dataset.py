"""
Script to prepare dataset for OCR model training
Creates train/val/test splits and validates data
"""

import os
import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def prepare_dataset(prescriptions_dir, lab_reports_dir, output_dir, 
                   train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    """
    Prepare dataset from prescription and lab report directories
    
    Args:
        prescriptions_dir: Path to prescriptions directory
        lab_reports_dir: Path to lab reports directory
        output_dir: Path to output directory
        train_split: Train set percentage
        val_split: Validation set percentage
        test_split: Test set percentage
        seed: Random seed
    """
    
    prescriptions_dir = Path(prescriptions_dir)
    lab_reports_dir = Path(lab_reports_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'texts').mkdir(parents=True, exist_ok=True)
    
    # Collect all samples
    all_samples = []
    
    # Prescriptions
    print("Processing prescriptions...")
    prescriptions_output = prescriptions_dir / 'Output'
    prescriptions_input = prescriptions_dir / 'Input'
    
    if prescriptions_output.exists():
        txt_files = list(prescriptions_output.glob('*.txt'))
        print(f"Found {len(txt_files)} prescription text files")
        
        # Check if corresponding images exist in Input directory
        for txt_file in txt_files:
            base_name = txt_file.stem
            # Try different image extensions
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                img_file = prescriptions_input / (base_name + ext)
                if img_file.exists():
                    all_samples.append({
                        'type': 'prescription',
                        'image_path': str(img_file),
                        'text_path': str(txt_file)
                    })
                    image_found = True
                    break
            
            if not image_found:
                print(f"Warning: No image found for {base_name}")
    
    # Lab Reports
    print("Processing lab reports...")
    lab_output = lab_reports_dir / 'Output'
    lab_input = lab_reports_dir / 'Input'
    
    if lab_output.exists():
        txt_files = list(lab_output.glob('*.txt'))
        print(f"Found {len(txt_files)} lab report text files")
        
        # Check if corresponding images exist in Input directory
        for txt_file in txt_files:
            base_name = txt_file.stem
            # Try different image extensions
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                img_file = lab_input / (base_name + ext)
                if img_file.exists():
                    all_samples.append({
                        'type': 'lab_report',
                        'image_path': str(img_file),
                        'text_path': str(txt_file)
                    })
                    image_found = True
                    break
            
            if not image_found:
                print(f"Warning: No image found for {base_name}")
    
    print(f"Total samples collected: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("ERROR: No samples found! Check your directory structure.")
        print(f"Prescriptions dir: {prescriptions_dir}")
        print(f"Lab reports dir: {lab_reports_dir}")
        return
    
    # Split data
    train_size = int(len(all_samples) * train_split)
    val_size = int(len(all_samples) * val_split)
    
    train_samples, temp_samples = train_test_split(
        all_samples, test_size=(1 - train_split), random_state=seed
    )
    
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=test_split / (val_split + test_split), random_state=seed
    )
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Copy files and create manifests
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    for split_name, samples in splits.items():
        manifest = []
        
        for idx, sample in enumerate(samples):
            try:
                # Copy image
                src_img = Path(sample['image_path'])
                dst_img = output_dir / split_name / 'images' / src_img.name
                
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"Warning: Image not found: {src_img}")
                    continue
                
                # Copy text
                src_txt = Path(sample['text_path'])
                dst_txt = output_dir / split_name / 'texts' / src_txt.name
                
                if src_txt.exists():
                    shutil.copy2(src_txt, dst_txt)
                else:
                    print(f"Warning: Text file not found: {src_txt}")
                    continue
                
                # Add to manifest
                manifest.append({
                    'image': str(dst_img.name),
                    'text_file': str(dst_txt.name),
                    'type': sample['type']
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
        
        # Save manifest
        manifest_path = output_dir / split_name / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved {len(manifest)} samples to {split_name} split")
    
    print(f"\nDataset preparation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OCR dataset")
    parser.add_argument(
        "--prescriptions-dir",
        default="../data/data1",
        help="Path to prescriptions directory"
    )
    parser.add_argument(
        "--lab-reports-dir",
        default="../data/lbmaske",
        help="Path to lab reports directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/processed",
        help="Path to output directory"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.prescriptions_dir,
        args.lab_reports_dir,
        args.output_dir,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed
    )
