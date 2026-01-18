import cv2
import pytesseract
import numpy as np
import argparse
import sys
import os

def extract_text_from_image(image_path):
    """
    Takes an image path, performs preprocessing, and returns extracted text.
    Optimized for noisy medical documents.
    """
    try:
        # 1. Check if file exists
        if not os.path.exists(image_path):
            return f"Error: File not found at {image_path}"

        # 2. Load the image
        img = cv2.imread(str(image_path))
        if img is None:
            return "Error: OpenCV could not read the image. Check format."

        # 3. Preprocessing (Stage 3: OCR Correction in PDF 3)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Rescale image to 300 DPI equivalent if it's too small
        height, width = gray.shape
        if width < 1000:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.medianBlur(gray, 3)
        
        # Adaptive Thresholding (Vital for varied paper quality)
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 4. Configure Tesseract 
        # --psm 3: Best for medical reports with headers and tables
        custom_config = r'--oem 3 --psm 3'

        # 5. Perform OCR
        text = pytesseract.image_to_string(thresh, config=custom_config)

        return text.strip()

    except Exception as e:
        return f"OCR Error: {str(e)}"

def main():
    # Setup Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="EMR OCR Extraction Tool")
    parser.add_argument("image", help="Path to the clinical image (PNG/JPG/PDF_page)")
    
    # Parse arguments
    args = parser.parse_args()

    # Run OCR
    result = extract_text_from_image(args.image)
    
    # Output to standard output (stdout)
    # This allows for piping to the next stage of your EMR workflow
    print(result)

if __name__ == "__main__":
    main()