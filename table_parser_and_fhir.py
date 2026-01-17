import re
import json
import uuid
import sys
import argparse
from datetime import datetime
from difflib import SequenceMatcher

class UniversalClinicalRegistry:
    """Stage 5: Comprehensive Registry for Tests, Vitals, and Meds"""
    def __init__(self):
        # Maps Stems/Synonyms to Canonical FHIR-ready entries
        self.registry = {
            # --- HEMATOLOGY (CBC) ---
            "HEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "stems": ["HEMO", "GLOBI", "HGB", "THLEMO", "HB"]},
            "WBC COUNT": {"code": "6690-2", "unit": "/uL", "stems": ["WBC", "WHITE", "LEUCO", "TOIL"]},
            "RBC COUNT": {"code": "2897-1", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "SUNT"]},
            "PLATELET": {"code": "777-3", "unit": "/uL", "stems": ["PLATE", "PLT", "PIATE"]},
            "PCV": {"code": "20570-8", "unit": "%", "stems": ["PCV", "HCT", "PACKED", "RACKED"]},
            "NEUTROPHILS": {"code": "770-8", "unit": "%", "stems": ["NEUT", "POLY"]},
            "LYMPHOCYTES": {"code": "731-0", "unit": "%", "stems": ["LYMP", "LYMPH"]},
            
            # --- RENAL (KFT) ---
            "CREATININE": {"code": "2160-0", "unit": "mg/dL", "stems": ["CREAT", "CREA"]},
            "UREA": {"code": "22664-7", "unit": "mg/dL", "stems": ["UREA"]},
            "URIC ACID": {"code": "2943-0", "unit": "mg/dL", "stems": ["URIC"]},
            
            # --- LIVER (LFT) ---
            "BILIRUBIN": {"code": "1975-2", "unit": "mg/dL", "stems": ["BILI", "BILIR"]},
            "SGOT": {"code": "1920-8", "unit": "U/L", "stems": ["SGOT", "AST"]},
            "SGPT": {"code": "1742-6", "unit": "U/L", "stems": ["SGPT", "ALT"]},
            
            # --- DIABETES & LIPIDS ---
            "GLUCOSE": {"code": "2339-0", "unit": "mg/dL", "stems": ["GLUC", "SUGAR", "FBS", "PPBS"]},
            "HBA1C": {"code": "4548-4", "unit": "%", "stems": ["HBA1C", "GLYCO"]},
            "CHOLESTEROL": {"code": "2093-3", "unit": "mg/dL", "stems": ["CHOL", "CHOLEST"]},
            
            # --- VITALS ---
            "BLOOD PRESSURE": {"code": "85354-9", "unit": "mmHg", "stems": ["BP", "SYS", "DIA", "PRESSURE"]},
            "PULSE": {"code": "8867-4", "unit": "bpm", "stems": ["PULSE", "HEART RATE", "HR"]},
            "SPO2": {"code": "2708-6", "unit": "%", "stems": ["SPO2", "SATURATION", "OXYGEN"]}
        }

class EMREngine:
    def __init__(self):
        self.registry = UniversalClinicalRegistry().registry

    def _clean(self, text):
        return re.sub(r'[^A-Z0-9\s\.\/]', '', text.upper())

    def _extract_numeric(self, line):
        """Advanced numeric extraction including BP patterns (120/80)"""
        # Search for BP format first
        bp_match = re.search(r"(\d{2,3}\/\d{2,3})", line)
        if bp_match:
            return bp_match.group(1)
        
        # Search for standard decimals, ignoring range patterns
        nums = re.findall(r"(\d+\.?\d*)", line)
        if not nums: return None
        # Return first significant number
        for n in nums:
            if len(n) > 1 or n in ["0", "1"]: return n
        return nums[0]

    def parse(self, raw_text):
        extracted = []
        lines = raw_text.split('\n')

        for line in lines:
            clean_line = self._clean(line)
            if len(clean_line) < 2: continue

            best_match = None
            for test, info in self.registry.items():
                if any(stem in clean_line for stem in info['stems']):
                    val = self._extract_numeric(line)
                    if val:
                        extracted.append({
                            "attribute": test,
                            "value": val,
                            "unit": info['unit'],
                            "loinc": info['code']
                        })
                        break 

        return self._to_fhir(extracted)

    def _to_fhir(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        seen = set()
        for item in data:
            uid = f"{item['attribute']}_{item['value']}"
            if uid in seen: continue
            seen.add(uid)
            
            entry = {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['loinc'], "display": item['attribute']}]},
                    "valueString" if '/' in str(item['value']) else "valueQuantity": 
                        {"value": float(item['value']), "unit": item['unit']} if '/' not in str(item['value']) else item['value']
                }
            }
            bundle["entry"].append(entry)
        return bundle

def main():
    parser = argparse.ArgumentParser(description="EMR Universal OCR Parser")
    parser.add_argument("input_file", nargs="?", type=argparse.FileType("r"), default=sys.stdin, 
                        help="Path to raw OCR text file (or pipe input)")
    args = parser.parse_args()

    raw_text = args.input_file.read()
    engine = EMREngine()
    fhir_bundle = engine.parse(raw_text)
    
    print(json.dumps(fhir_bundle, indent=2))

if __name__ == "__main__":
    main()