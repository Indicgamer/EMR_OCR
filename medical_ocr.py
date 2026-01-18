import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import sys
import json
import cv2
from paddleocr import PaddleOCR

class MedicalLayoutOCR:
    def __init__(self):
        # We enable use_gpu and keep ir_optim=False for stability
        self.ocr = PaddleOCR(
            lang='en', 
            use_angle_cls=True, 
            use_gpu=True,      # Changed to True for T4 GPU
            ir_optim=False,     # Still False to avoid the AnalysisConfig error
            show_log=False
        )

    def get_layout_rows(self, img_path):
        if not os.path.exists(img_path):
            return f"Error: File {img_path} not found"

        result = self.ocr.ocr(img_path, cls=True)
        
        if not result or result[0] is None:
            return ""

        blocks = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            y_center = (box[0][1] + box[2][1]) / 2
            blocks.append({"text": text, "y": y_center, "x": box[0][0]})

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

        final_text = ""
        for row in rows:
            row.sort(key=lambda b: b['x'])
            final_text += " ".join([b['text'] for b in row]) + "\n"
            
        return final_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python medical_ocr.py <image_path>")
    else:
        engine = MedicalLayoutOCR()
        print(engine.get_layout_rows(sys.argv[1]))