import cv2
import pytesseract
from PIL import Image
import numpy as np

def extract_text_from_image(image_path):
    """
    Takes an image path, performs preprocessing, and returns extracted text.
    Suitable for EMR integration.
    """
    try:
        # 1. Load the image
        img = cv2.imread(str(image_path))
        if img is None:
            return "Error: Could not read image."

        # 2. Preprocessing for Medical Docs (Grayscale -> Denoise -> Threshold)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Remove noise (helps with scans/faded prints)
        denoised = cv2.medianBlur(gray, 3)
        
        # Adaptive Thresholding (turns image into pure black and white)
        # This is vital for Tesseract to distinguish ink from paper
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 3. Configure Tesseract 
        # --psm 3: Fully automatic page segmentation (best for mixed layouts)
        # --oem 3: Default OCR engine mode
        custom_config = r'--oem 3 --psm 3'

        # 4. Perform OCR
        text = pytesseract.image_to_string(thresh, config=custom_config)

        return text.strip()

    except Exception as e:
        return f"OCR Error: {str(e)}"

# --- Simple usage for your workflow ---
if __name__ == "__main__":
    # Test on one of your images
    input_file = "D:\3rd Sem MTech\3rd SEM Projects\new_EMR\data\lbmaske\Input\BLR-0425-PA-0039192_E-PareshwarFinalBill_250427_1337@E.pdf_page_88.png" 
    result = extract_text_from_image(input_file)
    
    print("--- EXTRACTED TEXT ---")
    print(result)

