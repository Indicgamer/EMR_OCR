import re
import json
import requests
import uuid
from datetime import datetime
from difflib import get_close_matches

class ClinicalKnowledgeBase:
    """Stage 5: Local Clinical Knowledge Base (PDF 3 Standards)"""
    def __init__(self):
        # Comprehensive Local Dictionary (LOINC for Labs, SNOMED for Meds)
        self.anchors = {
            # HEMATOLOGY
            "HAEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "system": "LOINC"},
            "HEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "system": "LOINC"},
            "RBC COUNT": {"code": "2897-1", "unit": "mill/cmm", "system": "LOINC"},
            "WBC COUNT": {"code": "6690-2", "unit": "/ul", "system": "LOINC"},
            "PLATELET COUNT": {"code": "777-3", "unit": "/ul", "system": "LOINC"},
            "PCV": {"code": "20570-8", "unit": "%", "system": "LOINC"},
            "MCV": {"code": "30428-7", "unit": "fL", "system": "LOINC"},
            "MCH": {"code": "28539-4", "unit": "pg", "system": "LOINC"},
            "MCHC": {"code": "28540-2", "unit": "g/dL", "system": "LOINC"},
            "RDW": {"code": "788-0", "unit": "%", "system": "LOINC"},
            
            # BIOCHEMISTRY / LFT / KFT
            "BLOOD UREA": {"code": "22664-7", "unit": "mg/dL", "system": "LOINC"},
            "CREATININE": {"code": "2160-0", "unit": "mg/dL", "system": "LOINC"},
            "BILIRUBIN TOTAL": {"code": "1975-2", "unit": "mg/dL", "system": "LOINC"},
            "SGOT": {"code": "1920-8", "unit": "U/L", "system": "LOINC"},
            "SGPT": {"code": "1742-6", "unit": "U/L", "system": "LOINC"},
            "ALKALINE PHOSPHATASE": {"code": "6768-6", "unit": "U/L", "system": "LOINC"},
            
            # DIABETES / THYROID
            "HBA1C": {"code": "4548-4", "unit": "%", "system": "LOINC"},
            "FASTING BLOOD SUGAR": {"code": "1558-6", "unit": "mg/dL", "system": "LOINC"},
            "TSH": {"code": "11579-0", "unit": "uIU/mL", "system": "LOINC"},
        }

class ExternalTerminologyService:
    """Fetches official codes for terms not in local dictionary"""
    def get_loinc_code(self, term):
        try:
            # Using NLM Clinical Tables API (Free/Public)
            url = f"https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search?terms={term}"
            response = requests.get(url, timeout=5)
            data = response.json()
            if data[0] > 0:
                # Return top result: [Code, DisplayName]
                return {"code": data[3][0][0], "display": data[3][0][1]}
        except:
            return None
        return None

class ProfessionalEMRPipeline:
    def __init__(self):
        self.kb = ClinicalKnowledgeBase()
        self.api = ExternalTerminologyService()

    def process_ocr_to_fhir(self, ocr_text):
        extracted_data = []
        lines = ocr_text.split('\n')

        for line in lines:
            line_clean = line.strip().upper()
            if not line_clean: continue

            # 1. FUZZY ANCHOR MATCHING
            # We check if any word in the line is close to our local dictionary
            matched_test = None
            for anchor in self.kb.anchors.keys():
                if anchor in line_clean or get_close_matches(line_clean, [anchor], cutoff=0.6):
                    matched_test = anchor
                    break
            
            # 2. VALUE EXTRACTION (The Result)
            # Looks for numbers following symbols like ':', '>', '='
            # This ignores the reference range at the end of the line
            value_match = re.search(r"[:>=]\s*(\d+\.?\d*)", line_clean)
            
            if matched_test and value_match:
                info = self.kb.anchors[matched_test]
                extracted_data.append({
                    "test": matched_test,
                    "value": float(value_match.group(1)),
                    "code": info["code"],
                    "unit": info["unit"],
                    "system": "http://loinc.org"
                })
            
            # 3. EXTERNAL API FALLBACK
            # If no local match, but line looks like a test result
            elif value_match and len(line_clean.split()) > 1:
                potential_term = line_clean.split(':')[0].strip()
                api_result = self.api.get_loinc_code(potential_term)
                if api_result:
                    extracted_data.append({
                        "test": api_result["display"],
                        "value": float(value_match.group(1)),
                        "code": api_result["code"],
                        "unit": "check_report",
                        "system": "http://loinc.org"
                    })

        return self._generate_fhir_bundle(extracted_data)

    def _generate_fhir_bundle(self, data):
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
        for item in data:
            obs = {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": item["system"], "code": item["code"], "display": item["test"]}]},
                    "valueQuantity": {"value": item["value"], "unit": item["unit"]},
                    "effectiveDateTime": datetime.now().isoformat()
                }
            }
            bundle["entry"].append(obs)
        return bundle

# --- EXECUTION ---
ocr_input = """se / i oft S. Vadhavkar
/ é cc. (Micro). O FAL.
SHREE pal 27 row
AN OES |. / pAmoLogtcal
‘’ # COMPUTERISED
Tine 18.00 em to 8.00 pm * Sunday-8:00t0 12.00 r100n 7 Q yavesRon caves
ms it |
— tC —_ 7;
a: : Dr. BHAVESH CHAUHAN MD es 2 Po
: Shree Hospital IPD HR tine: N° 2:47:26
Sample Id : 10436879 Report Release Timo =: FN 13:42:13
COMPLETE BLOOD COUNT
Test Result Unit Biological Ref. Range
, © Haemoglobin > 9.10 (L) gmid! © 13.0-17.0 gnvdl
Total R.B.C. Count > 3.19 [L] millcmm 4.5-5.5 millicmm
Haematocrit (PCV/HCT) : 27.20 [L) %o 40.0-50.0 %
Mean Corpuscular Volume (M.C.V.) : 86.30 fl 83.0-95.0 fi
Mean Corpuscular Hb (M.C.H.) : 28.50 Pg 27.0-32.0 Pg
Mean Corpuscular Hb Cone (M.C.H.C.) : 33.50 g/dl 31.5-34.5 g/dl
{ #} Red cell Distribution Width (R.D.W.-: 16.5 [H] % 11.6-14.6 %
cv)
Total W.B.C. Count : 10560 [H] ful 4000-10000 /ul
DIFFERENTIAL COUNT:
Neutrophils : 87.7 [H) % 40-70 %
Lymphocytes : 5.9 [L} % 20-40 %
Eosinophils > 07 % 16%
Monocytes : «5.5 % 2-10 %
Basophils : 0.2 % 0-1 %
PLATELETS.""" # 'Random Sugar' is not in local dict, will trigger API

pipeline = ProfessionalEMRPipeline()
fhir_bundle = pipeline.process_ocr_to_fhir(ocr_input)

print(json.dumps(fhir_bundle, indent=2))