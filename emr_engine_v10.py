import re
import json
import uuid
import sys
from datetime import datetime
from thefuzz import fuzz, process

class ClinicalKnowledgeBase:
    """Stage 5: Comprehensive LOINC & Alias Registry"""
    def __init__(self):
        # Master Registry: LOINC -> Canonical Name -> Units -> Stems/Aliases
        self.registry = {
            "718-7":  {"display": "Hemoglobin", "unit": "g/dL", "stems": ["HEMO", "HGB", "HB", "HAEMO"]},
            "2897-1":  {"display": "RBC Count", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "SUNT"]},
            "6690-2":  {"display": "WBC Count", "unit": "/uL", "stems": ["WBC", "WHITE", "LEUCO", "TOIL"]},
            "777-3":   {"display": "Platelet Count", "unit": "/uL", "stems": ["PLATE", "PLT", "PIATE", "PIATO"]},
            "20570-8": {"display": "PCV / HCT", "unit": "%", "stems": ["PCV", "HCT", "HAEMATOCRIT", "PACKED CELL"]},
            "30428-7": {"display": "MCV", "unit": "fL", "stems": ["MCV", "BAG", "CELL VOLUME"]},
            "28539-4": {"display": "MCH", "unit": "pg", "stems": ["MCH", "CELI HEMO"]},
            "28540-2": {"display": "MCHC", "unit": "g/dL", "stems": ["MCHC", "CONC"]},
            "770-8":   {"display": "Neutrophils", "unit": "%", "stems": ["NEUT", "POLY"]},
            "731-0":   {"display": "Lymphocytes", "unit": "%", "stems": ["LYMPH"]},
            "711-2":   {"display": "Eosinophils", "unit": "%", "stems": ["EOSI", "ESINO"]},
            "742-7":   {"display": "Monocytes", "unit": "%", "stems": ["MONO"]},
            "706-2":   {"display": "Basophils", "unit": "%", "stems": ["BASO"]},
            "2160-0":  {"display": "Creatinine", "unit": "mg/dL", "stems": ["CREATININE", "CREA"]},
            "2339-0":  {"display": "Glucose", "unit": "mg/dL", "stems": ["GLUC", "SUGAR", "FBS", "PPBS"]}
        }
        self.all_stems = [s for k in self.registry for s in self.registry[k]['stems']]
        self.stem_to_code = {s: k for k in self.registry for s in self.registry[k]['stems']}

class UltimateParser:
    def __init__(self):
        self.kb = ClinicalKnowledgeBase()

    def _extract_numeric(self, segment):
        """Stage 3: OCR Correction. Fixes O/0 swaps and skips noise."""
        # 1. Strip bracketed noise like [H], [L], [4]
        segment = re.sub(r'\[.*?\]', ' ', segment)
        # 2. Fix O for 0 and I for 1
        segment = segment.replace('O', '0').replace('I', '1').replace('l', '1')
        # 3. Find numbers
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

            # Stage 4: Semantic Anchor Search
            # We look for the medical keyword in the line
            matched_stem = None
            words = re.split(r'[: ]', line_upper)
            for word in words:
                if len(word) < 2: continue
                # Fuzzy match each word against our medical stems
                match = process.extractOne(word, self.kb.all_stems, scorer=fuzz.ratio)
                if match and match[1] > 85: # 85% confidence
                    matched_stem = match[0]
                    break
            
            if matched_stem:
                loinc = self.kb.stem_to_code[matched_stem]
                # Extract value from the segment AFTER the stem
                payload = line_upper.split(matched_stem)[-1]
                value = self._extract_numeric(payload)
                
                if value and loinc not in extracted:
                    info = self.kb.registry[loinc]
                    extracted[loinc] = {
                        "code": loinc,
                        "display": info['display'],
                        "value": value,
                        "unit": info['unit']
                    }

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
    print(json.dumps(UltimateParser().parse(raw_ocr), indent=2))