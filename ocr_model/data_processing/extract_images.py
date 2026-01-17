"""
Extract images from PDF files (if needed)
Converts PDF pages to images for OCR training
"""

import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_dir, output_dir, dpi=300, fmt='png'):
    """
    Extract images from PDF files in a directory
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
        dpi: Resolution for PDF conversion
        fmt: Image format (png, jpg, etc.)
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir.glob('*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Convert PDF pages to images
            images = convert_from_path(str(pdf_file), dpi=dpi, fmt=fmt)
            
            # Save each page as image
            for page_num, image in enumerate(images, 1):
                output_name = f"{pdf_file.stem}_page_{page_num}.{fmt}"
                output_path = output_dir / output_name
                
                image.save(str(output_path), fmt.upper())
                logger.info(f"Saved: {output_name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
    
    logger.info(f"Extraction complete! Images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from PDFs")
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for extracted images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF conversion"
    )
    parser.add_argument(
        "--format",
        choices=['png', 'jpg', 'jpeg', 'bmp'],
        default='png',
        help="Output image format"
    )
    
    args = parser.parse_args()
    
    extract_images_from_pdf(
        args.pdf_dir,
        args.output_dir,
        args.dpi,
        args.format
    )
