import re
import json
import sys
import uuid
from datetime import datetime

class PRO_EMR_Engine:
    def __init__(self):
        # Expanded dictionary for General EMR
        self.registry = {
            "718-7":  {"display": "Hemoglobin", "stems": ["HEMO", "HGB", "HB"]},
            "6690-2":  {"display": "WBC Count", "stems": ["WBC", "WHITE", "LEUCO"]},
            "777-3":   {"display": "Platelet Count", "stems": ["PLATE", "PLT"]},
            "20570-8": {"display": "PCV / HCT", "stems": ["PCV", "HCT", "PACKED"]},
            "30428-7": {"display": "MCV", "stems": ["MCV"]},
            "2160-0":  {"display": "Creatinine", "stems": ["CREAT"]},
            "2339-0":  {"display": "Glucose", "stems": ["GLUC", "SUGAR", "FBS"]},
            "85354-9": {"display": "Blood Pressure", "stems": ["BP", "PRESSURE"]}
        }

    def parse_rows(self, row_text):
        entries = []
        lines = row_text.split('\n')
        
        for line in lines:
            line_upper = line.upper()
            
            for code, info in self.registry.items():
                if any(stem in line_upper for stem in info['stems']):
                    # Now we just look for a number in the SAME ROW
                    # This is 10x more accurate than Tesseract
                    num_match = re.search(r"(\d{2,3}\/\d{2,3}|\d+\.\d+|\d+)", line_upper)
                    if num_match:
                        val = num_match.group(1)
                        if val in ["2024", "2025"]: continue # Skip dates
                        
                        entries.append({
                            "display": info['display'],
                            "code": code,
                            "value": val
                        })
                        break # Stop searching for other tests on this row

        return self._generate_fhir(entries)

    def _generate_fhir(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now().isoformat(), "entry": []}
        for item in data:
            val_type = "valueString" if "/" in str(item['value']) else "valueQuantity"
            bundle["entry"].append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": item['code'], "display": item['display']}]},
                    val_type: {"value": float(item['value']) if val_type == "valueQuantity" else item['value'], "unit": "units"}
                }
            })
        return bundle

if __name__ == "__main__":
    raw_text = sys.stdin.read()
    print(json.dumps(PRO_EMR_Engine().parse_rows(raw_text), indent=2))