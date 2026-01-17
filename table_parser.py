import re
import json

class LabTableParser:
    def __init__(self):
        # Stage 3: Semantic Mapping (Based on PDF 2 & 4 Standards)
        self.loinc_map = {
            "HEMOGLOBIN": {"code": "718-7", "unit": "g/dL"},
            "PLATELET COUNT": {"code": "777-3", "unit": "10^3/uL"},
            "RBC COUNT": {"code": "2897-1", "unit": "10^6/uL"},
            "MCV": {"code": "30428-7", "unit": "fL"},
            "WBC": {"code": "6690-2", "unit": "/cumm"}
        }

    def parse_ocr_text(self, text):
        results = []
        lines = text.split('\n')
        
        for label, info in self.loinc_map.items():
            for line in lines:
                if label in line.upper():
                    # Regex to find all numbers in the line (handles decimals)
                    # This captures values from all 3 columns in your image
                    values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    
                    if values:
                        # We take the LAST value found on the line (the most recent column)
                        current_val = values[-1]
                        results.append({
                            "test": label,
                            "value": float(current_val),
                            "unit": info["unit"],
                            "loinc": info["code"]
                        })
        return results

# Test Stage 2 with your provided text
raw_ocr = """--- EXTRACTED TEXT ---
GE Te ve0! TENOR

PLATELET INDICES -
PLATELET COUNT

PLATELET INDICES -
Pow

PLATELET INDICES - MPV

PLATELET INDICES - P-
LCR

PLATELET INDICES - PCT

PSSR.MO

PSS W.MO

PERIPHERAL SMEAR
STUDY - PLATELETS

PERIPHERAL SMEAR
STUDY - PARASITE

PERIPHERAL SMEAR
STUDY - Note

HEMOGLOBIN

RBC COUNT

PACKED CELL
VOLUME(PCV)

RBC INDICES - MCV

RBC INDICES - MCH

RBC INDICES - MCHC
RBC INDICES - R.D.W.- SD
RBC INDICES - R.D.W.- CV

RBC INDICES - WBC

125000
/cumm

10.9 fL

10.6 f~L

30.0 %

0.13 %

Mild
Anisocytosi
s, Mild
Microcytosi
s, Mild
Hypochrom
ia, few
pencil cells
seen.
Within
Normal
Limits.

Mild
Reduction.

Not
detected

11.0 gm%

4.83
mil/cumm

35.3%

73.08 fL
22.77 pgm
31.16 g/dL
449

16.9

6220
fcumm

117000
{comm

13.7 ff

10.8 fi

32.3%

0.13 %

Mild
Anisocytosi
s, Mild
Microcytosi
s, Mild
Hypochrom
ia.

Mild
leucocytosis

Mild
Reduction.

Not
detected

Suggested -
Fe
supplement
. Close
Follow up
for Hb.

9.6 gm%

4.21
mil/eumm

30.2%

71.73 fl.
22.80 pgm
31.79 g/dL
43.0

16.4

12880
/cumm

209000
/cumm

12.2 fL

10.4 fL

28.3 %

0.22 %

Mild
Anisocytosi
s, Mild
Microcytosi
s, Mild
Hypochromi
a, Few
pencil cells
seen.

Mild
Poltymorpho
nuclear
Leucocytosi
s with mild
left shift.

Adequate

Not
detected

9.6 gm%

3.98
mil/eumm

29.5%

74.12
24.12 pgm
32.54 p/dL
43.4

16.8

14310
/cumm"""
parser = LabTableParser()
extracted_entities = parser.parse_ocr_text(raw_ocr)
print(json.dumps(extracted_entities, indent=2))