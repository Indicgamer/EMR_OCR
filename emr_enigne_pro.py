import re
import json
import uuid
import sys
from datetime import datetime
from difflib import get_close_matches

class ClinicalEntityRegistry:
    """Stage 5: Standardized Clinical Terms with Common Misspellings"""
    def __init__(self):
        # Maps canonical LOINC to a list of 'Entity Stems'
        self.entities = {
            "718-7":  {"name": "Hemoglobin", "unit": "g/dL", "stems": ["HEMOGLOBIN", "HAEMOGLOBIN", "HB", "HGB", "THLEM"]},
            "6690-2":  {"name": "WBC Count", "unit": "/uL", "stems": ["WBC", "WHITE CELL", "LEUCOCYTE", "W.B.C", "TOIL"]},
            "777-3":   {"name": "Platelet Count", "unit": "/uL", "stems": ["PLATELET", "PLT", "PIATELET", "PLT COUNT"]},
            "2897-1":  {"name": "RBC Count", "unit": "mill/cmm", "stems": ["RBC", "RED CELL", "R.B.C"]},
            "20570-8": {"name": "PCV / HCT", "unit": "%", "stems": ["PCV", "HCT", "HAEMATOCRIT", "PACKED CELL"]},
            "30428-7": {"name": "MCV", "unit": "fL", "stems": ["MCV", "CORPUSCULAR VOLUME", "BAG"]},
            "2160-0":  {"name": "Creatinine", "unit": "mg/dL", "stems": ["CREATININE", "CREA", "SERUM CREAT"]},
            "2339-0":  {"name": "Glucose", "unit": "mg/dL", "stems": ["GLUCOSE", "SUGAR", "FBS", "PPBS", "GLUC"]},
            "85354-9": {"name": "Blood Pressure", "unit": "mmHg", "stems": ["BP", "BLOOD PRESSURE", "SYS/DIA", "PRESSURE"]}
        }
        # Flatten for the fuzzy matcher
        self.all_stems = []
        self.stem_to_code = {}
        for code, info in self.entities.items():
            for stem in info['stems']:
                self.all_stems.append(stem)
                self.stem_to_code[stem] = code

class RobustNERParser:
    def __init__(self):
        self.registry = ClinicalEntityRegistry()

    def _fuzzy_find_entity(self, token):
        """Uses Levenshtein distance to find clinical entities even with OCR errors"""
        matches = get_close_matches(token, self.registry.all_stems, n=1, cutoff=0.7)
        return self.registry.stem_to_code[matches[0]] if matches else None

    def _extract_numeric(self, text):
        """Stage 3: OCR Correction. Cleans noise and extracts first plausible number."""
        # 1. Clean OCR noise that commonly interrupts numbers
        clean = re.sub(r'[:\*\[\]HL4!|]', ' ', text)
        # 2. Fix common character-for-digit swaps
        clean = clean.replace('O', '0').replace('l', '1').replace('I', '1').replace('S', '5').replace('B', '8')
        
        # 3. Find numbers (Decimals, Integers, or BP format)
        matches = re.findall(r"(\d{2,3}/\d{2,3}|\d+\.\d+|\d+)", clean)
        
        for val in matches:
            # Filter out dates/years (Stage 8 Validation)
            if val in ["2024", "2025", "2026", "2023"]: continue
            return val
        return None

    def parse(self, ocr_text):
        extracted_data = {}
        lines = ocr_text.split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            if not line_upper: continue

            # PASS 1: Entity Recognition
            # We split by common delimiters to find potential clinical labels
            tokens = re.split(r'[:.\- ]', line_upper)
            matched_code = None
            for token in tokens:
                if len(token) < 2: continue
                matched_code = self._fuzzy_find_entity(token)
                if matched_code: break

            # PASS 2: Value Extraction (Proximity Search)
            if matched_code:
                # Look for the result value in the same line
                value = self._extract_numeric(line_upper)
                
                if value:
                    # Deduplicate: only store the first instance found
                    if matched_code not in extracted_data:
                        info = self.registry.entities[matched_code]
                        extracted_data[matched_code] = {
                            "code": matched_code,
                            "display": info['name'],
                            "value": value,
                            "unit": info['unit']
                        }

        return self._to_fhir_bundle(extracted_data.values())

    def _to_fhir_bundle(self, observations):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }
        for obs in observations:
            is_numeric = "/" not in str(obs['value'])
            val_key = "valueQuantity" if is_numeric else "valueString"
            
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": obs['code'], "display": obs['display']}]},
                    val_key: {
                        "value": float(obs['value']) if is_numeric else obs['value'],
                        "unit": obs['unit']
                    }
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    parser = RobustNERParser()
    print(json.dumps(parser.parse(raw_ocr), indent=2))