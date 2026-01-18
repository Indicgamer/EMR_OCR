import re
import json
import uuid
import sys
from datetime import datetime

class OCRHealer:
    """Stage 3: Heals character-to-number swaps common in noisy medical OCR"""
    @staticmethod
    def fix_value(val_str):
        # OCR often swaps numbers for look-alike letters
        swaps = {'B': '8', 'G': '6', 'S': '5', 'O': '0', 'I': '1', 'l': '1', 'A': '4'}
        for char, num in swaps.items():
            val_str = val_str.replace(char, num)
        
        # Extract only the numeric part
        clean = re.sub(r'[^0-9\.]', '', val_str)
        
        # Logic for missing decimals (e.g., '452' -> '45.2' for PCV)
        if clean.isdigit() and len(clean) == 3 and not clean.startswith('0'):
             # If it's a 3-digit number like 452 or 116, it's likely a missing decimal
             return f"{clean[:2]}.{clean[2:]}"
        return clean

class MasterRegistry:
    """Stage 5: Maps garbled medical terms to standard LOINC codes"""
    def __init__(self):
        self.data = {
            "718-7":  {"display": "Hemoglobin", "stems": ["HEMO", "THLEM", "HGB", "HB"]},
            "20570-8": {"display": "PCV / HCT", "stems": ["PCV", "HCT", "RACKED", "PACKED", "VOLUME"]},
            "2897-1":  {"display": "RBC Count", "stems": ["RBC", "SUNT", "RED CELL"]},
            "30428-7": {"display": "MCV", "stems": ["MCV", "CELL VOLUME", "BAG"]},
            "28539-4": {"display": "MCH", "stems": ["MCHY", "CELI HEMO", "MCH "]},
            "28540-2": {"display": "MCHC", "stems": ["MCHC", "CONC"]},
            "6690-2":  {"display": "WBC Count", "stems": ["WBC", "TOIL", "WHITE"]},
            "770-8":   {"display": "Neutrophils", "stems": ["NEUT", "POLY"]},
            "731-0":   {"display": "Lymphocytes", "stems": ["LYMPH"]},
            "711-2":   {"display": "Eosinophils", "stems": ["ESINO", "EOSI"]},
            "742-7":   {"display": "Monocytes", "stems": ["MONO"]},
            "777-3":   {"display": "Platelet Count", "stems": ["PLATE", "PIATE"]}
        }

class FinalEMRPipeline:
    def __init__(self):
        self.registry = MasterRegistry().data
        self.healer = OCRHealer()

    def parse(self, text):
        fhir_entries = []
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper()
            
            # Find the best clinical anchor
            matched_code = None
            for code, info in self.registry.items():
                if any(stem in line_upper for stem in info['stems']):
                    matched_code = code
                    break
            
            if matched_code:
                # Extract numeric part (Stage 3)
                # Look for sequences of digits/letters that look like numbers
                raw_val_match = re.search(r"(\d+[\.\,]?\d*|[A-Z]*\d+[A-Z]*\d*)", line_upper)
                if raw_val_match:
                    raw_val = raw_val_match.group(1)
                    clean_val = self.healer.fix_value(raw_val)
                    
                    try:
                        val_float = float(clean_val)
                        # Sanity check: ignore common OCR noise/years
                        if val_float in [2025.0, 1.0, 0.0]: continue
                        
                        fhir_entries.append(self._create_obs(matched_code, val_float))
                    except:
                        continue

        return self._build_bundle(fhir_entries)

    def _create_obs(self, code, value):
        info = self.registry[code]
        return {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": info['display']}]},
                "valueQuantity": {"value": value, "unit": "units"}
            }
        }

    def _build_bundle(self, entries):
        # De-duplicate by code (keep only the first occurrence)
        unique_entries = []
        codes_seen = set()
        for entry in entries:
            c = entry['resource']['code']['coding'][0]['code']
            if c not in codes_seen:
                unique_entries.append(entry)
                codes_seen.add(c)
        
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": unique_entries
        }

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(FinalEMRPipeline().parse(raw_ocr), indent=2))