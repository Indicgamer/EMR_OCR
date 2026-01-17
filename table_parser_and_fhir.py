import re
import json
import uuid
from datetime import datetime
from difflib import get_close_matches

class MedicalKnowledgeBase:
    """Stage 5 (PDF 3): Medical Dictionary with Synonyms and Codes"""
    def __init__(self):
        # We map all possible OCR variations to a single 'Canonical Name'
        self.test_registry = {
            "HAEMOGLOBIN": {"code": "718-7", "unit": "g/dL", "aliases": ["HB", "HEMOGLOBIN", "HAEMOGLOB", "HEMOGLO"]},
            "RBC COUNT": {"code": "2897-1", "unit": "mill/cmm", "aliases": ["TOTAL RBC", "RED CELL", "TOTAL RBC COUNT"]},
            "WBC COUNT": {"code": "6690-2", "unit": "/ul", "aliases": ["TOTAL WBC", "WHITE CELL", "WBC", "LEUCOCYTES"]},
            "PCV": {"code": "20570-8", "unit": "%", "aliases": ["HAEMATOCRIT", "HCT", "PACKED CELL VOLUME"]},
            "MCV": {"code": "30428-7", "unit": "fL", "aliases": ["MEAN CORPUSCULAR VOLUME"]},
            "MCH": {"code": "28539-4", "unit": "pg", "aliases": ["MEAN CORPUSCULAR HB"]},
            "MCHC": {"code": "28540-2", "unit": "g/dL", "aliases": ["MEAN CORPUSCULAR HB CONE", "M.C.H.C."]},
            "RDW": {"code": "788-0", "unit": "%", "aliases": ["RED CELL DISTRIBUTION", "RDWCV", "RDWSD"]},
            "PLATELET COUNT": {"code": "777-3", "unit": "/ul", "aliases": ["PIATOLET", "PLT", "PLATELETS"]},
            "NEUTROPHILS": {"code": "770-8", "unit": "%", "aliases": ["NEUTRO", "NEUT"]},
            "LYMPHOCYTES": {"code": "731-0", "unit": "%", "aliases": ["LYMPHO", "LYMPH"]},
        }
        
        # Flattened list for fast fuzzy matching
        self.all_keywords = []
        self.keyword_to_canonical = {}
        for canonical, info in self.test_registry.items():
            self.all_keywords.append(canonical)
            self.keyword_to_canonical[canonical] = canonical
            for alias in info['aliases']:
                self.all_keywords.append(alias)
                self.keyword_to_canonical[alias] = canonical

class DeepNormalizer:
    """Stage 3 (PDF 3): OCR Correction and Text Cleaning"""
    
    @staticmethod
    def heal_text(text):
        """Fixes symbols and common OCR character swaps in labels"""
        # 1. Remove clinical junk/noise
        text = re.sub(r'[\{\}\#\@\_\!\(\)\[\]\>]', '', text)
        
        # 2. Fix common character swaps (OCR 'leetspeak')
        text = text.replace('0', 'O')  # 0 to O in words
        text = text.replace('5', 'S')  # 5 to S in words
        text = text.replace('|', 'I')  # | to I
        
        return text.strip().upper()

    @staticmethod
    def heal_value(value_str):
        """Ensures the extracted result is a clean number"""
        # Fix common number swaps: O -> 0, I -> 1
        value_str = value_str.upper().replace('O', '0').replace('I', '1')
        # Remove any non-numeric characters except decimal
        value_str = re.sub(r'[^0-9\.]', '', value_str)
        return value_str

class AdvancedEMRPipeline:
    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.normalizer = DeepNormalizer()

    def process(self, ocr_text):
        extracted_data = []
        lines = ocr_text.split('\n')

        for line in lines:
            # 1. Separate Label and Value (Targeting the first numeric result)
            # This regex captures the label part and the first numeric part
            parts = re.split(r'[:>=]', line)
            if len(parts) < 2: continue
            
            raw_label = parts[0]
            raw_value = parts[1]

            # 2. Heal the Label
            healed_label = self.normalizer.heal_text(raw_label)
            
            # 3. Fuzzy Search for the correct test
            # Snaps "HAEMOGLOBI" to "HAEMOGLOBIN" or "HB" to "HAEMOGLOBIN"
            best_match = get_close_matches(healed_label, self.kb.all_keywords, n=1, cutoff=0.5)
            
            if best_match:
                canonical_name = self.kb.keyword_to_canonical[best_match[0]]
                info = self.kb.test_registry[canonical_name]
                
                # 4. Heal the Value
                # Looks for the first number in the value part
                value_match = re.search(r"(\d+[\.\,]?\d*)", raw_value)
                if value_match:
                    clean_val = self.normalizer.heal_value(value_match.group(1))
                    
                    extracted_data.append({
                        "test": canonical_name,
                        "value": float(clean_val),
                        "code": info["code"],
                        "unit": info["unit"]
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
                    "code": {"coding": [{"system": "http://loinc.org", "code": item["code"], "display": item["test"]}]},
                    "valueQuantity": {"value": item["value"], "unit": item["unit"]},
                    "effectiveDateTime": datetime.now().isoformat()
                }
            }
            bundle["entry"].append(obs)
        return bundle

# --- EXECUTION ---
ocr_messy_input = """
--- EXTRACTED TEXT ---
Test Report
K.P. Patil Building,
Near Shivaji Maharaj Statue,
Mohopada, Tal. Khalapur,
Olst. Raigad - 410 222.

Ue RTL

 

\ SHREE DIAGNOSTIC

    
 
     
       
 
 
     

Pati mC 5 :
Y | : i Geader sMale

.

HBB doctor. 7-Apr-2025 10-1 AM
ao |

|

i, oon & ¢ OUNT C c ne ra

tHlemoptohin

  

11.6 ginfdl 130-17.0

racked Cell VY, . ~y

6 Packed Coll Volume CHCT) 452 %, 40-50
RB -

sunt 5.10 mill/emm BGG
Mean Cell Volume( MCV) BAG n WS-10T
‘Mean Celi Hemoplobin( MCHY 22.7 pr 27-33
Mean Ceil Hb Conc( MCHC) 25.7 xy 32-38
Toil WBC Count 9200 cells/euram 4000-11000
Differential % WBCs count
Neutrophils 66 yy 30-70
Lymphocytes 24 : XY, 20-40
t i esinophits 4 % LG

PI, recy
Monocytes 06 % 0-10
PO ey
Absolute Differential Count:
Absolute Neutrophils Count 6072 feumm 2000-7000
Absolute Lymphocytes Count 2208 /Joumm 1000-3000
Absolute Monocytes Count 952 / cute 200-1000
Absolute Eosinophils Count, AEC 30H /eunm 20-500
Platelet Count 398000 cetls/cumm FY
Ponte ec La pectence

ENO OF REPORT

Scan to Valtdate £5 .

Or. Swapnil V.Sirmukaddam
M.0 (#athalogy)

  

purpose.
ton with clinical Sndeng

 

reco gree
"""

pipeline = AdvancedEMRPipeline()
final_fhir = pipeline.process(ocr_messy_input)

print(json.dumps(final_fhir, indent=2))