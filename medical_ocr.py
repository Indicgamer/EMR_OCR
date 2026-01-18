from paddleocr import PaddleOCR
import json
import sys

class MedicalLayoutOCR:
    def __init__(self):
        # use_angle_cls=True helps with tilted mobile photos of reports
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def get_layout_rows(self, img_path):
        result = self.ocr.ocr(img_path, cls=True)
        
        # Flatten the result and add mid-point Y coordinates for row-grouping
        blocks = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            y_center = (box[0][1] + box[2][1]) / 2  # Calculate middle of the text block
            blocks.append({"text": text, "y": y_center, "x": box[0][0]})

        # Group blocks into rows if their Y-coordinates are within 15 pixels of each other
        # This is the "Magic" that fixes the table parsing
        blocks.sort(key=lambda b: b['y'])
        rows = []
        if not blocks: return ""

        current_row = [blocks[0]]
        for i in range(1, len(blocks)):
            if abs(blocks[i]['y'] - blocks[i-1]['y']) < 15: # 15px threshold for a 'row'
                current_row.append(blocks[i])
            else:
                rows.append(current_row)
                current_row = [blocks[i]]
        rows.append(current_row)

        # Sort each row horizontally (Left to Right)
        final_text = ""
        for row in rows:
            row.sort(key=lambda b: b['x'])
            row_text = " ".join([b['text'] for b in row])
            final_text += row_text + "\n"
            
        return final_text

if __name__ == "__main__":
    engine = MedicalLayoutOCR()
    print(engine.get_layout_rows(sys.argv[1]))