import re
import json
import uuid
import sys
import requests
from datetime import datetime
from thefuzz import fuzz, process

class EnsembleParser:
    def __init__(self):
        # 1. Local Knowledge Base (Fast Track)
        self.local_registry = {
            "718-7": ["HEMOGLOBIN", "HB", "HGB", "HAEMOGLOBIN"],
            "6690-2": ["WBC", "WHITE CELL", "LEUCOCYTE"],
            "777-3": ["PLATELET", "PLT", "PIATELET"],
            "20570-8": ["PCV", "HCT", "HAEMATOCRIT"],
            "2339-0": ["GLUCOSE", "SUGAR", "FBS", "PPBS"]
        }
        self.local_vocab = [item for sublist in self.local_registry.values() for item in sublist]

    def _query_external_medical_api(self, term):
        """
        Stage 5: External Coding Fallback (PDF 3)
        Queries NLM Clinical Tables for LOINC mapping.
        """
        try:
            # Search LOINC for the unknown term
            url = f"https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search?terms={term}&max_results=1"
            response = requests.get(url, timeout=2).json()
            if response[0] > 0:
                loinc_code = response[3][0][0]
                display_name = response[3][0][1]
                return {"code": loinc_code, "display": display_name}
        except:
            pass
        return None

    def _extract_value(self, line):
        """Standardized Extraction for results and BP"""
        # Cleanup OCR noise
        line = re.sub(r'[:\*\[\]HL4!]', ' ', line)
        # Matches BP (120/80), Decimals (11.6), and Large Ints (370)
        matches = re.findall(r"(\d{2,3}/\d{2,3}|\d+\.\d+|\d+)", line)
        for m in matches:
            if m in ["2024", "2025", "2026"]: continue
            return m
        return None

    def parse(self, ocr_text):
        final_data = {}
        lines = ocr_text.split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            if len(line_upper) < 4: continue

            # --- STEP 1: LOCAL FUZZY MATCH ---
            match_results = process.extract(line_upper, self.local_vocab, scorer=fuzz.partial_ratio, limit=1)
            
            target_code = None
            target_display = ""

            if match_results and match_results[0][1] > 85:
                # Found locally
                matched_term = match_results[0][0]
                for code, stems in self.local_registry.items():
                    if matched_term in stems:
                        target_code = code
                        target_display = matched_term
                        break
            
            # --- STEP 2: EXTERNAL API FALLBACK ---
            # If local match failed, we try to extract the 'Label' part of the line and ask NLM
            else:
                potential_label = re.sub(r'[^A-Z ]', '', line_upper).strip()
                if len(potential_label) > 4:
                    external_res = self._query_external_medical_api(potential_label)
                    if external_res:
                        target_code = external_res['code']
                        target_display = external_res['display']

            # --- STEP 3: VALUE PAIRING ---
            if target_code:
                value = self._extract_value(line_upper)
                if value and target_code not in final_data:
                    final_data[target_code] = {
                        "code": target_code,
                        "display": target_display,
                        "value": value
                    }

        return self._to_fhir(final_data.values())

    def _to_fhir(self, observations):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        for obs in observations:
            is_bp = "/" in str(obs['value'])
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": obs['code'], "display": obs['display']}]},
                    "valueQuantity" if not is_bp else "valueString": {
                        "value": float(obs['value']) if not is_bp else obs['value'],
                        "unit": "units"
                    }
                }
            })
        return bundle

if __name__ == "__main__":
    raw_ocr = sys.stdin.read()
    print(json.dumps(EnsembleParser().parse(raw_ocr), indent=2))