import re
import json
import uuid
import sys
from datetime import datetime
from thefuzz import fuzz

class MedicalOntology:
    """Stage 5: Standardized LOINC Mapping with expanded Hematology & Biochem"""
    def __init__(self):
        self.registry = {
            "718-7":  {"display": "Hemoglobin", "unit": "g/dL", "stems": ["HEMOGLOBIN", "HB", "HGB", "HAEMOGLOBIN", "THLEM"]},
            "2897-1":  {"display": "RBC Count", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "SUNT"]},
            "6690-2":  {"display": "WBC Count", "unit": "/uL", "stems": ["WBC", "WHITE CELL", "W.B.C", "TOIL"]},
            "777-3":   {"display": "Platelet Count", "unit": "/uL", "stems": ["PLATELET", "PLT", "PIATELET"]},
            "20570-8": {"display": "PCV / HCT", "unit": "%", "stems": ["PCV", "HCT", "HEMATOCRIT", "HAEMATOCRIT", "RACKED"]},
            "30428-7": {"display": "MCV", "unit": "fL", "stems": ["MCV", "M.C.V", "CELL VOLUME", "BAG"]},
            "28539-4": {"display": "MCH", "unit": "pg", "stems": ["MCH", "M.C.H", "CELI HEMO"]},
            "28540-2": {"display": "MCHC", "unit": "g/dL", "stems": ["MCHC", "M.C.H.C", "CONC"]},
            "770-8":   {"display": "Neutrophils", "unit": "%", "stems": ["NEUTROPHILS", "NEUT", "POLY"]},
            "731-0":   {"display": "Lymphocytes", "unit": "%", "stems": ["LYMPHOCYTES", "LYMPH"]},
            "711-2":   {"display": "Eosinophils", "unit": "%", "stems": ["EOSINOPHILS", "EOSI"]},
            "742-7":   {"display": "Monocytes", "unit": "%", "stems": ["MONOCYTES", "MONO"]},
            "706-2":   {"display": "Basophils", "unit": "%", "stems": ["BASOPHILS", "BASO"]},
            "2339-0":  {"display": "Glucose", "unit": "mg/dL", "stems": ["GLUCOSE", "SUGAR", "FBS", "PPBS"]}
        }

class ValueShieldParser:
    def __init__(self):
        self.ontology = MedicalOntology().registry

    def _smart_extract_value(self, payload):
        """
        Stage 3: Advanced Numeric Extraction
        1. Shields (deletes) the reference range.
        2. Strips clinical flags [H], [L], [4].
        3. Returns the actual patient result.
        """
        # A. Shield Reference Ranges: Remove anything like '13.0 - 17.0' or '4.5-5.5'
        # We look for number-hyphen-number and delete that whole block
        payload = re.sub(r'\d+\.?\d*\s*-\s*\d+\.?\d*', ' ', payload)
        
        # B. Strip Clinical Flags & Noise: Remove [H], [L], [4], (*), {#}
        payload = re.sub(r'\[.*?\]', ' ', payload)
        payload = re.sub(r'\{.*?\}', ' ', payload)
        payload = re.sub(r'\(.*?\)', ' ', payload)
        payload = re.sub(r'[:\*#]', ' ', payload)
        
        # C. Heal common OCR errors in digits
        payload = payload.replace('O', '0').replace('I', '1').replace('l', '1')

        # D. Capture the Result: Usually the first number left in the cleaned payload
        # Supports decimals (9.10) and large integers (10560)
        matches = re.findall(r"(\d+\.\d+|\d+)", payload)
        
        for val in matches:
            # Skip report years/dates
            if val in ["2024", "2025", "2026"]: continue
            return val
        return None

    def parse(self, text):
        extracted = {}
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            if len(line_upper) < 4: continue

            for loinc, info in self.ontology.items():
                best_ratio = 0
                matched_stem = ""
                
                # Semantic match to find the test name
                for stem in info['stems']:
                    ratio = fuzz.partial_ratio(stem, line_upper)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        matched_stem = stem
                
                if best_ratio > 85:
                    # Isolate the text to the right of the test name
                    parts = line_upper.split(matched_stem)
                    if len(parts) > 1:
                        payload = parts[-1]
                        value = self._smart_extract_value(payload)
                        
                        if value and loinc not in extracted:
                            extracted[loinc] = {
                                "code": loinc,
                                "display": info['display'],
                                "value": value,
                                "unit": info['unit']
                            }
                    break # Match found for this line

        return self._to_fhir(extracted.values())

    def _to_fhir(self, items):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        for item in items:
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation", "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    "valueQuantity": {"value": float(item['value']), "unit": item['unit']}
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(ValueShieldParser().parse(raw_ocr), indent=2))