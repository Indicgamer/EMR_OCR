import re
import json
import uuid
import sys
import requests
from datetime import datetime

class ClinicalValidator:
    """Rejects values that are clinically impossible (e.g., Hemoglobin > 100)"""
    ranges = {
        "718-7": {"max": 25, "min": 2},     # Hemoglobin
        "2897-1": {"max": 10, "min": 1},    # RBC
        "20570-8": {"max": 70, "min": 10},  # PCV/HCT
        "30428-7": {"max": 150, "min": 50}, # MCV
        "28539-4": {"max": 50, "min": 10},  # MCH
        "6690-2": {"max": 100000, "min": 100}, # WBC
    }

    @staticmethod
    def is_plausible(loinc, value):
        if loinc not in ClinicalValidator.ranges: return True
        r = ClinicalValidator.ranges[loinc]
        return r["min"] <= value <= r["max"]

class MedicalTermResolver:
    """Uses NLM API to resolve garbled OCR into official medical terms"""
    def __init__(self):
        self.cache = {}
        # Local fallback for speed and high-confidence garbles
        self.local_fixes = {
            "THLEMOPTOHIN": "HEMOGLOBIN", "HLEMO": "HEMOGLOBIN",
            "RACKED": "PCV", "PACKED": "PCV", "VOLUME": "PCV",
            "CELI": "CELL", "MCHY": "MCH", "MCHC": "MCHC",
            "SUNT": "RBC COUNT", "TOIL": "WBC COUNT"
        }

    def resolve(self, term):
        term = term.strip().upper()
        # 1. Quick Local Fix
        for typo, fix in self.local_fixes.items():
            if typo in term: return fix
        
        # 2. NLM API Lookup (Stage 5 Normalization)
        if term in self.cache: return self.cache[term]
        try:
            url = f"https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search?terms={term}"
            res = requests.get(url, timeout=2).json()
            if res[0] > 0:
                resolved = res[3][0][1].upper()
                self.cache[term] = resolved
                return resolved
        except: pass
        return term

class AdvancedEMREngine:
    def __init__(self):
        self.resolver = MedicalTermResolver()
        self.validator = ClinicalValidator()
        self.loinc_map = {
            "HEMOGLOBIN": "718-7", "PCV": "20570-8", "RBC COUNT": "2897-1",
            "MCV": "30428-7", "MCH": "28539-4", "MCHC": "28540-2",
            "WBC COUNT": "6690-2", "PLATELET": "777-3", "NEUTROPHILS": "770-8"
        }

    def parse(self, text):
        results = {}
        lines = text.split('\n')

        for line in lines:
            # Stage 3: Clean line
            clean_line = re.sub(r'[^A-Z0-9\s\.\/\-\:]', '', line.upper())
            
            # Find the value first (numeric sequence)
            # This regex avoids picking up dates by requiring context
            val_match = re.search(r"(?<!\d)(?<!\-)(?<!\/)(\d{1,5}\.\d{1,2}|\d{2,6})(?!\d)(?!\-)(?!\/)", line)
            
            if val_match:
                potential_val = float(val_match.group(1))
                # Get text to the left of the value
                label_part = line[:val_match.start()].strip()
                if len(label_part) < 2: continue

                resolved_term = self.resolver.resolve(label_part)
                
                # Find best matching LOINC
                for key, code in self.loinc_map.items():
                    if key in resolved_term:
                        # Clinical Validation (Stop '2025' from becoming 'Hemoglobin')
                        if self.validator.is_plausible(code, potential_val):
                            # Store only the best/first result to prevent duplicates
                            if code not in results:
                                results[code] = {
                                    "display": key,
                                    "value": potential_val,
                                    "code": code
                                }
                        break

        return self._to_fhir(results.values())

    def _to_fhir(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        for item in data:
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    "valueQuantity": {"value": item['value'], "unit": "units"}
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(AdvancedEMREngine().parse(raw_ocr), indent=2))