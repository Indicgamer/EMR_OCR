import re
import json
import uuid
import sys
from datetime import datetime
from thefuzz import fuzz, process

class MedicalOntology:
    """Stage 5: Standardized Ontology (aligned with ABDM/FHIR)"""
    def __init__(self):
        # Master Registry: LOINC Code -> Canonical Name -> Aliases
        self.registry = {
            "718-7":  {"name": "Hemoglobin", "unit": "g/dL", "aliases": ["HB", "HGB", "HAEMOGLOBIN", "HEMOPTOHIN"]},
            "6690-2":  {"name": "WBC Count", "unit": "/uL", "aliases": ["WBC", "WHITE BLOOD CELL", "TOTAL LEUCOCYTES"]},
            "777-3":   {"name": "Platelet Count", "unit": "/uL", "aliases": ["PLT", "PLATELETS", "PIATELET"]},
            "20570-8": {"name": "PCV / HCT", "unit": "%", "aliases": ["HEMATOCRIT", "PACKED CELL VOLUME", "PCV"]},
            "30428-7": {"name": "MCV", "unit": "fL", "aliases": ["MEAN CORPUSCULAR VOLUME", "M.C.V."]},
            "2160-0":  {"name": "Creatinine", "unit": "mg/dL", "aliases": ["SERUM CREATININE", "CREA"]},
            "2339-0":  {"name": "Glucose", "unit": "mg/dL", "aliases": ["SUGAR", "FASTING BLOOD GLUCOSE", "FBS"]},
            "85354-9": {"name": "Blood Pressure", "unit": "mmHg", "aliases": ["BP", "B.P.", "SYSTOLIC/DIASTOLIC"]}
        }
        # Flat list for semantic matching
        self.search_vocab = []
        self.alias_to_code = {}
        for code, info in self.registry.items():
            names = [info['name']] + info['aliases']
            self.search_vocab.extend(names)
            for name in names:
                self.alias_to_code[name] = code

class SemanticEMRParser:
    def __init__(self):
        self.ontology = MedicalOntology()

    def _normalize_value(self, raw_val):
        """Stage 3: OCR Correction & Healing"""
        # Remove common OCR symbols
        clean = re.sub(r'[^\d\.\/]', '', raw_val)
        # Fix trailing dots or accidental commas
        clean = clean.strip('.')
        return clean

    def parse(self, ocr_text):
        final_observations = {}
        lines = ocr_text.split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            if not line_upper or len(line_upper) < 3: continue

            # Stage 4: Semantic NER
            # We look for the best medical match in the line using Fuzzy Wuzzy
            # This handles 'H3moglob1n' or 'H.B' errors
            match_results = process.extract(line_upper, self.ontology.search_vocab, 
                                           scorer=fuzz.partial_ratio, limit=1)
            
            if match_results and match_results[0][1] > 80: # 80% confidence threshold
                best_alias = match_results[0][0]
                loinc_code = self.ontology.alias_to_code[best_alias]
                
                # Stage 3: Contextual Value Extraction
                # Find numbers near the match or at the end of the line
                # Supports decimals and Blood Pressure formats
                numbers = re.findall(r"(\d{2,3}/\d{2,3}|\d+\.\d+|\d+)", line_upper)
                
                valid_value = None
                for n in numbers:
                    if n in ["2024", "2025", "2026"]: continue
                    valid_value = self._normalize_value(n)
                    break
                
                if valid_value:
                    # Stage 5: Coding & FHIR Structuring
                    if loinc_code not in final_observations:
                        meta = self.ontology.registry[loinc_code]
                        final_observations[loinc_code] = {
                            "code": loinc_code,
                            "display": meta['name'],
                            "value": valid_value,
                            "unit": meta['unit']
                        }

        return self._to_fhir_bundle(final_observations.values())

    def _to_fhir_bundle(self, data):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }
        for item in data:
            is_bp = "/" in str(item['value'])
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    "valueQuantity" if not is_bp else "valueString": {
                        "value": float(item['value']) if not is_bp else item['value'],
                        "unit": item['unit']
                    }
                }
            })
        return bundle

if __name__ == "__main__":
    ocr_input = sys.stdin.read()
    print(json.dumps(SemanticEMRParser().parse(ocr_input), indent=2))