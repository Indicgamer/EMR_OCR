import re
import json
import uuid
import sys
from datetime import datetime
from thefuzz import fuzz

class MedicalOntology:
    """Stage 5: Exhaustive Alias Registry for Indian Labs"""
    def __init__(self):
        # Maps LOINC -> Canonical Name -> List of all possible OCR variations
        self.registry = {
            "718-7":  {"display": "Hemoglobin", "unit": "g/dL", "stems": ["HEMOGLOBIN", "HB", "HGB", "HAEMOGLOBIN", "THLEM"]},
            "2897-1":  {"display": "RBC Count", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "R.B.C", "SUNT"]},
            "6690-2":  {"display": "WBC Count", "unit": "/uL", "stems": ["WBC", "WHITE CELL", "LEUCOCYTE", "W.B.C", "TOIL"]},
            "777-3":   {"display": "Platelet Count", "unit": "/uL", "stems": ["PLATELET", "PLT", "PIATELET", "PLT COUNT"]},
            "20570-8": {"display": "PCV / HCT", "unit": "%", "stems": ["PCV", "HCT", "HEMATOCRIT", "PACKED CELL", "HAEMATOCRIT", "RACKED"]},
            "30428-7": {"display": "MCV", "unit": "fL", "stems": ["MCV", "M.C.V", "CELL VOLUME", "CORPUSCULAR VOLUME", "BAG"]},
            "28539-4": {"display": "MCH", "unit": "pg", "stems": ["MCH", "M.C.H", "CORPUSCULAR HB", "CELI HEMO"]},
            "28540-2": {"display": "MCHC", "unit": "g/dL", "stems": ["MCHC", "M.C.H.C", "CONC", "CONCENTRATION"]},
            "770-8":   {"display": "Neutrophils", "unit": "%", "stems": ["NEUTROPHILS", "NEUT", "POLY", "NEUTRO"]},
            "731-0":   {"display": "Lymphocytes", "unit": "%", "stems": ["LYMPHOCYTES", "LYMPH", "LYMPHO"]},
            "711-2":   {"display": "Eosinophils", "unit": "%", "stems": ["EOSINOPHILS", "EOSI", "ESINO"]},
            "742-7":   {"display": "Monocytes", "unit": "%", "stems": ["MONOCYTES", "MONO"]},
            "706-2":   {"display": "Basophils", "unit": "%", "stems": ["BASOPHILS", "BASO"]},
            "2160-0":  {"display": "Creatinine", "unit": "mg/dL", "stems": ["CREATININE", "CREA", "SERUM CREAT"]},
            "2339-0":  {"display": "Glucose", "unit": "mg/dL", "stems": ["GLUCOSE", "SUGAR", "FBS", "PPBS", "GLUC"]}
        }

class SemanticParserV11:
    def __init__(self):
        self.ontology = MedicalOntology().registry

    def _extract_numeric(self, segment):
        """Stage 3: OCR Correction & Numeric Extraction"""
        # Remove brackets [H], [L], [4] and noise
        segment = re.sub(r'\[.*?\]', ' ', segment)
        segment = re.sub(r'[:\*#}{]', ' ', segment)
        # Heal OCR swaps
        segment = segment.replace('O', '0').replace('I', '1').replace('l', '1')
        
        # Find first numeric result (decimal or int)
        matches = re.findall(r"(\d+\.\d+|\d+)", segment)
        for m in matches:
            if m in ["2024", "2025", "2026"]: continue
            return m
        return None

    def parse(self, text):
        extracted = {}
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            if len(line_upper) < 4: continue

            # Stage 4: Semantic N-Gram Search
            # For every clinical test we know, check if it's hidden in this line
            for loinc, info in self.ontology.items():
                best_ratio = 0
                matched_stem = ""
                
                for stem in info['stems']:
                    # partial_ratio finds the stem even if it's merged: e.g. "MeanCorpuscular"
                    ratio = fuzz.partial_ratio(stem, line_upper)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        matched_stem = stem
                
                # If we have a high-confidence semantic match (cutoff 85%)
                if best_ratio > 85:
                    # Find the value in the segment AFTER the match
                    payload = line_upper.split(matched_stem)[-1]
                    value = self._extract_numeric(payload)
                    
                    if value and loinc not in extracted:
                        extracted[loinc] = {
                            "code": loinc,
                            "display": info['display'],
                            "value": value,
                            "unit": info['unit']
                        }

        return self._to_fhir(extracted.values())

    def _to_fhir(self, items):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }
        for item in items:
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    "valueQuantity": {"value": float(item['value']), "unit": item['unit']}
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(SemanticParserV11().parse(raw_ocr), indent=2))