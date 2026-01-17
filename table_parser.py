import re
import json
import uuid
from datetime import datetime

class PreciseMedicalPipeline:
    def __init__(self):
        # Official LOINC Mapping (PDF 4 Standards)
        # We include common OCR variants found in your text
        self.clinical_map = {
            "HAEMOGLOBIN": "718-7",
            "R.B.C. COUNT": "2897-1",
            "HAEMATOCRIT": "20570-8",
            "M.C.V.": "30428-7",
            "M.C.H.": "28539-4",
            "M.C.H.C.": "28540-2",
            "W.B.C. COUNT": "6690-2",
            "PLATELET COUNT": "777-3",
            "NEUTROPHILS": "770-8",
            "LYMPHOCYTES": "731-0"
        }

    def stage_2_extract(self, ocr_text):
        extracted_data = []
        lines = ocr_text.split('\n')

        for line in lines:
            line_upper = line.upper()
            
            # Check for anchors in each line
            for label, loinc in self.clinical_map.items():
                if label in line_upper:
                    # REGEX: Find the first decimal or integer following the label
                    # We ignore ranges like 13.0-17.0 by stopping at the first match
                    match = re.search(r"[:>]\s*(\d+\.?\d*)", line)
                    
                    if match:
                        value = match.group(1)
                        extracted_data.append({
                            "test": label,
                            "loinc": loinc,
                            "value": float(value),
                            "unit": self._detect_unit(line)
                        })
                    break # Move to next line once a label is found
        
        return extracted_data

    def _detect_unit(self, line):
        if 'G/DL' in line.upper() or 'GM/DL' in line.upper(): return "g/dL"
        if '%' in line: return "%"
        if 'FL' in line.upper(): return "fL"
        if 'PG' in line.upper(): return "pg"
        return "units"

    def stage_3_fhir(self, data):
        """Stage 7 of PDF 3: Integration into EHR (FHIR)"""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }

        for item in data:
            observation = {
                "resource": {
                    "resourceType": "Observation",
                    "id": str(uuid.uuid4())[:8],
                    "status": "final",
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": item["loinc"], "display": item["test"]}]
                    },
                    "valueQuantity": {
                        "value": item["value"],
                        "unit": item["unit"]
                    },
                    "effectiveDateTime": datetime.now().isoformat()
                }
            }
            bundle["entry"].append(observation)
        
        return bundle

# --- EXECUTION ---
ocr_output = """se / i oft S. Vadhavkar
/ é cc. (Micro). O FAL.
SHREE pal 27 row
AN OES |. / pAmoLogtcal
‘’ # COMPUTERISED
Tine 18.00 em to 8.00 pm * Sunday-8:00t0 12.00 r100n 7 Q yavesRon caves
ms it |
— tC —_ 7;
a: : Dr. BHAVESH CHAUHAN MD es 2 Po
: Shree Hospital IPD HR tine: N° 2:47:26
Sample Id : 10436879 Report Release Timo =: FN 13:42:13
COMPLETE BLOOD COUNT
Test Result Unit Biological Ref. Range
, © Haemoglobin > 9.10 (L) gmid! © 13.0-17.0 gnvdl
Total R.B.C. Count > 3.19 [L] millcmm 4.5-5.5 millicmm
Haematocrit (PCV/HCT) : 27.20 [L) %o 40.0-50.0 %
Mean Corpuscular Volume (M.C.V.) : 86.30 fl 83.0-95.0 fi
Mean Corpuscular Hb (M.C.H.) : 28.50 Pg 27.0-32.0 Pg
Mean Corpuscular Hb Cone (M.C.H.C.) : 33.50 g/dl 31.5-34.5 g/dl
{ #} Red cell Distribution Width (R.D.W.-: 16.5 [H] % 11.6-14.6 %
cv)
Total W.B.C. Count : 10560 [H] ful 4000-10000 /ul
DIFFERENTIAL COUNT:
Neutrophils : 87.7 [H) % 40-70 %
Lymphocytes : 5.9 [L} % 20-40 %
Eosinophils > 07 % 16%
Monocytes : «5.5 % 2-10 %
Basophils : 0.2 % 0-1 %
PLATELETS."""

pipeline = PreciseMedicalPipeline()
# 1. Extract
structured_entities = pipeline.stage_2_extract(ocr_output)
# 2. FHIR-ize
fhir_json = pipeline.stage_3_fhir(structured_entities)

print(json.dumps(fhir_json, indent=2))