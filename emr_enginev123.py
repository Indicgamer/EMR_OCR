import re
import json
import uuid
import sys
from datetime import datetime
from thefuzz import fuzz

class ClinicalAtlas:
    """Stage 5: Standardized Mapping for Indian Lab Layouts"""
    def __init__(self):
        self.registry = {
            "718-7":  {"display": "Hemoglobin", "unit": "gm/dl", "stems": ["HAEMOGLOBIN", "HB", "HGB", "HEMOGLOBIN"]},
            "2897-1":  {"display": "RBC Count", "unit": "mill/cmm", "stems": ["R.B.C", "RED CELL", "RBC"]},
            "20570-8": {"display": "PCV / HCT", "unit": "%", "stems": ["HAEMATOCRIT", "PCV", "HCT", "HEMATOCRIT"]},
            "30428-7": {"display": "MCV", "unit": "fL", "stems": ["M.C.V", "MCV", "CORPUSCULAR VOLUME"]},
            "28539-4": {"display": "MCH", "unit": "pg", "stems": ["M.C.H", "MCH", "CORPUSCULAR HB"]},
            "28540-2": {"display": "MCHC", "unit": "g/dl", "stems": ["M.C.H.C", "MCHC", "CONC"]},
            "6690-2":  {"display": "WBC Count", "unit": "/uL", "stems": ["W.B.C", "WBC", "WHITE CELL"]},
            "777-3":   {"display": "Platelet Count", "unit": "/uL", "stems": ["PLATELET", "PLT", "PLATELETS"]},
            "770-8":   {"display": "Neutrophils", "unit": "%", "stems": ["NEUTROPHILS", "NEUT", "POLY"]},
            "731-0":   {"display": "Lymphocytes", "unit": "%", "stems": ["LYMPHOCYTES", "LYMPH"]},
            "711-2":   {"display": "Eosinophils", "unit": "%", "stems": ["EOSINOPHILS", "EOSI"]},
            "742-7":   {"display": "Monocytes", "unit": "%", "stems": ["MONOCYTES", "MONO"]},
            "706-2":   {"display": "Basophils", "unit": "%", "stems": ["BASOPHILS", "BASO"]}
        }

class ClinicalProximityParser:
    def __init__(self):
        self.atlas = ClinicalAtlas().registry

    def _clean_and_extract(self, line):
        """
        Stage 3 & 4: Cleans the line and finds the most plausible clinical result.
        """
        # 1. Remove Reference Ranges (e.g., 13.0-17.0)
        # This is vital to prevent picking the range instead of the result
        line = re.sub(r'\d+\.?\d*\s*-\s*\d+\.?\d*', ' ', line)
        
        # 2. Remove Bracketed Flags like [H], [L], [4]
        line = re.sub(r'\[.*?\]', ' ', line)
        
        # 3. Heal OCR digit errors
        line = line.replace('O', '0').replace('I', '1').replace('l', '1')

        # 4. Find all numbers (decimals and integers)
        # In a lab report line, the first number remaining is almost always the result
        matches = re.findall(r"(\d+\.\d+|\d+)", line)
        
        for val in matches:
            # Skip noise like dates or very small isolated indices
            if val in ["2024", "2025", "2026", "0", "1"]: continue
            return val
        return None

    def parse(self, text):
        extracted = {}
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            if len(line_upper) < 5: continue

            # Trigger check: Does this line look like one of our clinical tests?
            best_match_code = None
            highest_score = 0
            
            for loinc, info in self.atlas.items():
                for stem in info['stems']:
                    # Use fuzzy matching to see if the test name is in the line
                    score = fuzz.partial_ratio(stem, line_upper)
                    if score > 85 and score > highest_score:
                        highest_score = score
                        best_match_code = loinc
            
            if best_match_code:
                # If triggered, extract the value from the WHOLE line
                # But we clean it first to ensure we don't get the range
                val = self._clean_and_extract(line_upper)
                
                if val:
                    info = self.atlas[best_match_code]
                    # Deduplicate: only the first result for each code
                    if best_match_code not in extracted:
                        extracted[best_match_code] = {
                            "code": best_match_code,
                            "display": info['display'],
                            "value": val,
                            "unit": info['unit']
                        }

        return self._build_bundle(extracted.values())

    def _build_bundle(self, results):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }
        for res in results:
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": res['code'], "display": res['display']}]},
                    "valueQuantity": {
                        "value": float(res['value']),
                        "unit": res['unit']
                    }
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(ClinicalProximityParser().parse(raw_ocr), indent=2))