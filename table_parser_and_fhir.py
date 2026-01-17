import re
import json
import uuid
from datetime import datetime
from difflib import SequenceMatcher

class RobustMedicalDictionary:
    """Stage 5: Canonical Dictionary with Error-Prone Stems"""
    def __init__(self):
        # We define 'Stems' (unique parts of words) to identify tests even if garbled
        self.registry = {
            "HEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "stems": ["HEMO", "GLOBI", "HGB", "HLEMO"]},
            "WBC COUNT": {"code": "6690-2", "unit": "cells/cumm", "stems": ["WBC", "WHITE", "LEUCO", "TOIL"]},
            "RBC COUNT": {"code": "2897-1", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "SUNT"]},
            "PLATELET COUNT": {"code": "777-3", "unit": "cells/cumm", "stems": ["PLATE", "PLT", "PIATE"]},
            "PCV": {"code": "20570-8", "unit": "%", "stems": ["PCV", "HCT", "PACKED", "RACKED"]},
            "MCV": {"code": "30428-7", "unit": "fL", "stems": ["MCV", "CELL VOL"]},
            "MCH": {"code": "28539-4", "unit": "pg", "stems": ["MCH", "CELI HEMO"]},
            "NEUTROPHILS": {"code": "770-8", "unit": "%", "stems": ["NEUT", "POLY"]},
            "LYMPHOCYTES": {"code": "731-0", "unit": "%", "stems": ["LYMP", "LYMPH"]},
            "EOSINOPHILS": {"code": "711-2", "unit": "%", "stems": ["EOSI", "ESINO"]},
            "MONOCYTES": {"code": "742-7", "unit": "%", "stems": ["MONO"]}
        }

class RobustEMRExtractor:
    def __init__(self):
        self.db = RobustMedicalDictionary()

    def _normalize(self, text):
        """Removes noise and heals common character swaps"""
        text = text.upper()
        text = re.sub(r'[^A-Z0-9\s\.]', '', text) # Remove junk symbols
        return text

    def _get_similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def _extract_value(self, line_text):
        """
        Intelligent Value Picker:
        Finds the most likely 'Patient Result' by ignoring typical reference range patterns.
        """
        # Find all numbers (including decimals)
        nums = re.findall(r"(\d+\.?\d*)", line_text)
        if not nums: return None
        
        # Heuristic: The result is usually the first number that isn't a single digit noise
        for n in nums:
            if len(n) > 1 or n in ["0", "1"]: # Keep 0 or 1 for counts, skip single digit noise
                return n
        return nums[0]

    def process_ocr(self, raw_text):
        extracted = []
        lines = raw_text.split('\n')

        for line in lines:
            clean_line = self._normalize(line)
            if len(clean_line) < 3: continue

            matched_test = None
            highest_score = 0

            # 1. Scoring Logic: Check against stems and canonical names
            for test, info in self.db.registry.items():
                # Check stems
                for stem in info['stems']:
                    if stem in clean_line:
                        score = 0.8 # Base score for stem match
                        # Increase score if the line is also fuzzy-similar to the test name
                        score += (self._get_similarity(clean_line, test) * 0.2)
                        
                        if score > highest_score:
                            highest_score = score
                            matched_test = test

            # 2. Threshold Check (Must be at least 70% confident)
            if matched_test and highest_score > 0.7:
                val = self._extract_value(line)
                if val:
                    data = self.db.registry[matched_test]
                    extracted.append({
                        "test": matched_test,
                        "value": float(val),
                        "code": data["code"],
                        "unit": data["unit"]
                    })

        return self._generate_fhir(extracted)

    def _generate_fhir(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
        seen = set() # Prevent duplicates from 'Absolute' sections
        for item in data:
            if item['test'] in seen: continue
            seen.add(item['test'])
            
            obs = {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item["code"], "display": item["test"]}]},
                    "valueQuantity": {"value": item["value"], "unit": item["unit"]},
                    "effectiveDateTime": datetime.now().isoformat()
                }
            }
            bundle["entry"].append(obs)
        return bundle

# --- EXECUTION ---
ocr_noise = """
tHlemoptohin 11.6 ginfdl 130-17.0
racked Cell VY 45.2 % 40-50
SUNT 5.10 mill/emm BGG
Toil WBC Count 9200 cells/euram
Neutrophils 66 yy 30-70
PIATELET COUNT 398000
"""

pipeline = RobustEMRExtractor()
fhir_json = pipeline.process_ocr(ocr_noise)

print(json.dumps(fhir_json, indent=2))