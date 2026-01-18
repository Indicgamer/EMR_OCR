import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
from paddleocr import PaddleOCR
import sys
import json

class MedicalLayoutOCR:
    def __init__(self):
        # Initializing PaddleOCR
        # We use a smaller model for speed in Colab
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def get_layout_rows(self, img_path):
        # Perform OCR
        result = self.ocr.ocr(img_path, cls=True)
        
        if not result or result[0] is None:
            return ""

        blocks = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            # Get the center Y coordinate to group into rows
            y_center = (box[0][1] + box[2][1]) / 2
            blocks.append({"text": text, "y": y_center, "x": box[0][0]})

        # Grouping logic (15px threshold)
        blocks.sort(key=lambda b: b['y'])
        rows = []
        if not blocks: return ""

        current_row = [blocks[0]]
        for i in range(1, len(blocks)):
            if abs(blocks[i]['y'] - blocks[i-1]['y']) < 15:
                current_row.append(blocks[i])
            else:
                rows.append(current_row)
                current_row = [blocks[i]]
        rows.append(current_row)

        # Build output string
        final_text = ""
        for row in rows:
            row.sort(key=lambda b: b['x']) # Sort Left-to-Right
            row_text = " ".join([b['text'] for b in row])
            final_text += row_text + "\n"
            
        return final_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python medical_ocr_pro.py <image_path>")
    else:
        engine = MedicalLayoutOCR()
        print(engine.get_layout_rows(sys.argv[1]))