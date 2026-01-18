import re
import json
import uuid
import sys
import argparse
from datetime import datetime

class ClinicalAtlas:
    """Universal mapping of medical terms to LOINC codes"""
    def __init__(self):
        self.map = {
            "718-7":  {"display": "Hemoglobin", "stems": ["HEMO", "HGB", "HB", "THLEMO", "HAEMO"]},
            "2897-1":  {"display": "RBC Count", "stems": ["RBC", "RED CELL", "SUNT"]},
            "6690-2":  {"display": "WBC Count", "stems": ["WBC", "WHITE", "LEUCO", "TOIL"]},
            "777-3":   {"display": "Platelet Count", "stems": ["PLATE", "PLT", "PIATE", "PIATO"]},
            "20570-8": {"display": "PCV / HCT", "stems": ["PCV", "HCT", "RACKED", "PACKED", "VOLUME"]},
            "30428-7": {"display": "MCV", "stems": ["MCV", "BAG", "CELL VOLUME"]},
            "28539-4": {"display": "MCH", "stems": ["MCH", "CELI HEMO", "MCHY"]},
            "28540-2": {"display": "MCHC", "stems": ["MCHC", "CONC"]},
            "770-8":   {"display": "Neutrophils", "stems": ["NEUT", "POLY"]},
            "731-0":   {"display": "Lymphocytes", "stems": ["LYMPH"]},
            "711-2":   {"display": "Eosinophils", "stems": ["EOSI", "ESINO"]},
            "742-7":   {"display": "Monocytes", "stems": ["MONO"]},
            "2160-0":  {"display": "Creatinine", "stems": ["CREAT"]},
            "22664-7": {"display": "Urea", "stems": ["UREA"]},
            "2339-0":  {"display": "Glucose", "stems": ["GLUC", "SUGAR", "FBS", "PPBS"]},
            "85354-9": {"display": "Blood Pressure", "stems": ["BP", "PRESSURE", "SYS", "DIA"]},
            "8867-4":  {"display": "Pulse", "stems": ["PULSE", "HEART RATE", "HR"]},
            "2708-6":  {"display": "SpO2", "stems": ["SPO2", "OXYGEN", "SATURATION"]}
        }

class GeneralEMREngine:
    def __init__(self):
        self.atlas = ClinicalAtlas().map

    def _heal_numeric(self, val_str):
        """Fixes OCR character swaps and handles missing decimals"""
        swaps = {'B': '8', 'G': '6', 'S': '5', 'O': '0', 'I': '1', 'l': '1', 'A': '4'}
        for char, num in swaps.items():
            val_str = val_str.replace(char, num)
        
        # Extract numeric part
        clean = re.sub(r'[^0-9\.\/]', '', val_str)
        
        # Handle 3-digit missing decimals (e.g. 116 -> 11.6 or 452 -> 45.2)
        if clean.replace('.', '').isdigit() and len(clean) == 3 and not clean.startswith('0'):
            if '.' not in clean:
                return f"{clean[:2]}.{clean[2:]}"
        return clean

    def parse(self, raw_text):
        text_upper = raw_text.upper()
        
        # 1. Locate all "Clinical Anchors" in the text
        anchors = []
        for code, info in self.atlas.items():
            for stem in info['stems']:
                for m in re.finditer(re.escape(stem), text_upper):
                    anchors.append({"code": code, "display": info['display'], "start": m.start()})

        # 2. Locate all "Numeric Values" in the text
        numbers = []
        # Matches BP (120/80), Decimals (11.6), and Integers (9200)
        num_pattern = r"(\d{2,3}\/\d{2,3}|\d+\.\d+|\d+)"
        for m in re.finditer(num_pattern, text_upper):
            val = m.group(1)
            # Skip common years to reduce noise
            if val in ["2024", "2025", "2026"]: continue
            numbers.append({"val": self._heal_numeric(val), "start": m.start()})

        # 3. Proximity Matching Engine
        # For every number found, find the closest anchor that appears BEFORE it
        clinical_results = {}
        for num in numbers:
            closest_anchor = None
            min_dist = 150 # Max character distance to look back
            
            for anchor in anchors:
                dist = num['start'] - anchor['start']
                if 0 < dist < min_dist:
                    min_dist = dist
                    closest_anchor = anchor
            
            if closest_anchor:
                # Deduplicate: only keep the first value found for each test
                code = closest_anchor['code']
                if code not in clinical_results:
                    clinical_results[code] = {
                        "display": closest_anchor['display'],
                        "code": code,
                        "value": num['val']
                    }

        return self._to_fhir(clinical_results.values())

    def _to_fhir(self, data):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }
        for item in data:
            val_type = "valueString" if "/" in str(item['value']) else "valueQuantity"
            entry = {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    val_type: {"value": float(item['value']) if val_type == "valueQuantity" else item['value']}
                }
            }
            # Add units if it's a quantity
            if val_type == "valueQuantity":
                entry["resource"][val_type]["unit"] = "units"

            bundle["entry"].append(entry)
        return bundle

def main():
    parser = argparse.ArgumentParser(description="Universal EMR Proximity Parser")
    parser.add_argument("input", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args()
    
    raw_text = args.input.read()
    engine = GeneralEMREngine()
    print(json.dumps(engine.parse(raw_text), indent=2))

if __name__ == "__main__":
    main()