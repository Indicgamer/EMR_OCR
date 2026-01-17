from transformers import pipeline
import re
import json

class SimpleMedicalNER:
    def __init__(self):
        # Load a lightweight, high-speed Biomedical NER model
        print("[*] Initializing NER Engine...")
        self.ner_model = pipeline(
            "ner", 
            model="d4data/biomedical-ner-all", 
            aggregation_strategy="simple"
        )
        
        # Simple fallback mapping for LOINC codes (Stage 5 PDF 3)
        self.loinc_codes = {
            "HEMOGLOBIN": "718-7",
            "RBC COUNT": "2897-1",
            "WBC": "6690-2",
            "PLATELET COUNT": "777-3"
        }

    def extract(self, text):
        # 1. Run NER on the text
        # This identifies 'Diagnostic_procedure' (Tests) and 'Lab_value' (Numbers)
        entities = self.ner_model(text)
        
        extracted_results = []
        
        # 2. Structural Parsing logic for your specific tabular text
        # Because Tesseract separates labels and values, we find labels and then 
        # look for the nearest numbers.
        
        # Extract labels found by NER
        labels = [e['word'].upper() for e in entities if e['entity_group'] in ['Diagnostic_procedure', 'Detailed_description']]
        # Extract values using a simple Regex for speed (Stage 3 PDF 3)
        values = re.findall(r"[-+]?\d*\.\d+|\d+", text)

        # Match labels to values (based on the order they appear in your OCR output)
        # Your OCR output lists labels first, then values.
        for i in range(min(len(labels), len(values))):
            test_name = labels[i]
            # Clean common OCR noise from labels
            test_name = re.sub(r'[^A-Z\s]', '', test_name).strip()
            
            if len(test_name) > 2: # Ignore tiny noise strings
                extracted_results.append({
                    "test": test_name,
                    "value": values[i],
                    "loinc": self.loinc_codes.get(test_name, "30428-7"), # Default to MCV if unknown
                    "unit": "standard"
                })

        return extracted_results

# --- EXECUTION ---
# Using the specific OCR output you provided
raw_text = """
--- EXTRACTED TEXT ---
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
/cumm
"""

extractor = SimpleMedicalNER()
structured_data = extractor.extract(raw_text)

print("\n--- STAGE 2: NER EXTRACTED DATA ---")
print(json.dumps(structured_data, indent=2))