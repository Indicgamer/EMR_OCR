import re
import json
import uuid
import sys
import requests
from datetime import datetime
from difflib import get_close_matches

class ClinicalKnowledgeBase:
    """Stage 5: Canonical Dictionary with Synonyms and Codes"""
    def __init__(self):
        # Maps all possible variations to a Canonical Name and LOINC
        self.registry = {
            "HEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "aliases": ["HB", "HGB", "THLEMOPTOHIN", "HEMOPTOHIN"]},
            "PCV": {"code": "20570-8", "unit": "%", "aliases": ["HCT", "HAEMATOCRIT", "PACKED CELL", "RACKED CELL"]},
            "RBC COUNT": {"code": "2897-1", "unit": "mill/cmm", "aliases": ["RED CELL", "SUNT", "TOTAL RBC"]},
            "WBC COUNT": {"code": "6690-2", "unit": "/uL", "aliases": ["TOTAL WBC", "WHITE CELL", "LEUCOCYTES", "TOIL"]},
            "MCV": {"code": "30428-7", "unit": "fL", "aliases": ["MEAN CELL VOLUME", "BAG"]},
            "MCH": {"code": "28539-4", "unit": "pg", "aliases": ["MEAN CELL HB", "CELI HEMO", "MCHY"]},
            "MCHC": {"code": "28540-2", "unit": "g/dL", "aliases": ["M.C.H.C.", "CONC"]},
            "PLATELET": {"code": "777-3", "unit": "/uL", "aliases": ["PLT", "PIATELET", "PLATELET COUNT"]},
            "NEUTROPHILS": {"code": "770-8", "unit": "%", "aliases": ["NEUT", "POLY"]},
            "LYMPHOCYTES": {"code": "731-0", "unit": "%", "aliases": ["LYMPH"]},
            "BLOOD PRESSURE": {"code": "85354-9", "unit": "mmHg", "aliases": ["BP", "SYS/DIA"]},
            "CREATININE": {"code": "2160-0", "unit": "mg/dL", "aliases": ["CREA", "CREAT"]},
            "GLUCOSE": {"code": "2339-0", "unit": "mg/dL", "aliases": ["SUGAR", "FBS", "PPBS", "GLUC"]}
        }
        self.all_keywords = []
        self.key_to_canonical = {}
        for canonical, info in self.registry.items():
            self.all_keywords.append(canonical)
            self.key_to_canonical[canonical] = canonical
            for alias in info['aliases']:
                self.all_keywords.append(alias)
                self.key_to_canonical[alias] = canonical

class EMRHealer:
    """Stage 3: Fixing character-to-digit swaps in OCR output"""
    @staticmethod
    def fix_val(val_str):
        swaps = {'B': '8', 'G': '6', 'S': '5', 'O': '0', 'I': '1', 'l': '1', 'A': '4'}
        # Only swap if it looks like a corrupted number (e.g., BAG or 9.1O)
        if any(c in swaps for c in val_str):
            for c, n in swaps.items(): val_str = val_str.replace(c, n)
        clean = re.sub(r'[^0-9\.\/]', '', val_str)
        # Fix missing decimals in 3-digit results (e.g., 452 -> 45.2)
        if clean.isdigit() and len(clean) == 3 and not clean.startswith('0'):
            return f"{clean[:2]}.{clean[2:]}"
        return clean

class ExternalCodeService:
    """Stage 5: Real-time search for unknown medical terms"""
    def search(self, term):
        try:
            url = f"https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search?terms={term}"
            res = requests.get(url, timeout=2).json()
            if res[0] > 0:
                return {"code": res[3][0][0], "display": res[3][0][1]}
        except: return None
        return None

class UltimateEMREngine:
    def __init__(self):
        self.kb = ClinicalKnowledgeBase()
        self.api = ExternalCodeService()
        self.healer = EMRHealer()

    def parse(self, row_text):
        final_results = {}
        lines = row_text.split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            if not line_upper: continue

            # 1. FUZZY MATCHING (Local Dictionary + Normalization)
            matched_canonical = None
            # Find best match in line
            possible = get_close_matches(line_upper, self.kb.all_keywords, n=1, cutoff=0.5)
            if possible:
                matched_canonical = self.kb.key_to_canonical[possible[0]]
            
            # 2. EXTRACT VALUE FROM SAME ROW
            # Handles BP (120/80) and standard decimals
            val_match = re.search(r"(\d{2,3}\/\d{2,3}|\d+[A-Z\.]+\d*|\d+\.\d+|\d+)", line_upper)
            
            if val_match:
                raw_val = val_match.group(1)
                clean_val = self.healer.fix_val(raw_val)
                
                if matched_canonical:
                    info = self.kb.registry[matched_canonical]
                    self._add_result(final_results, matched_canonical, info['code'], clean_val, info['unit'])
                
                # 3. EXTERNAL API FALLBACK (If it's a number but we don't recognize the text)
                else:
                    label_part = line_upper.split(raw_val)[0].strip()
                    if len(label_part) > 3:
                        api_res = self.api.search(label_part)
                        if api_res:
                            self._add_result(final_results, api_res['display'], api_res['code'], clean_val, "units")

        return self._generate_fhir(final_results.values())

    def _add_result(self, results, display, code, value, unit):
        # Clinical Validation: Ignore dates and impossible years
        try:
            v_float = float(value) if '/' not in str(value) else 0
            if v_float in [2024.0, 2025.0, 2026.0]: return
        except: pass
        
        # Deduplicate by code
        if code not in results:
            results[code] = {"display": display, "code": code, "value": value, "unit": unit}

    def _generate_fhir(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        for item in data:
            val_type = "valueString" if "/" in str(item['value']) else "valueQuantity"
            entry = {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    val_type: {"value": float(item['value']) if val_type == "valueQuantity" else item['value'], "unit": item['unit']}
                }
            }
            bundle["entry"].append(entry)
        return bundle

if __name__ == "__main__":
    raw_text = sys.stdin.read()
    engine = UltimateEMREngine()
    print(json.dumps(engine.parse(raw_text), indent=2))