import cv2
import pytesseract
import uuid
import json
from datetime import datetime
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.bundle import Bundle, BundleEntry

class CoreFHIRPipeline:
    def __init__(self):
        self.tess_config = r'--oem 3 --psm 6'

    def extract_and_convert(self, image_path):
        # --- STAGE 1: OCR ---
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarization improves OCR accuracy for medical docs
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        raw_text = pytesseract.image_to_string(thresh, config=self.tess_config)

        # --- STAGE 2:            DATA PARSING (Simplified for MWE) ---
        # In production, use Regex or NLP to extract these from raw_text
        extracted = {
            "patient_name": "Aditya Sharma",
            "lab_test": "Hemoglobin",
            "lab_result": 14.2,
            "unit": "g/dL",
            "drug": "Amlodipine 5mg",
            "snomed_code": "319709001" # SNOMED for Amlodipine
        }

        # --- STAGE 3: FHIR RESOURCE GENERATION ---
        
        # 1. Create Patient Resource
        patient = Patient()
        patient.id = str(uuid.uuid4())
        patient.name = [{"text": extracted["patient_name"]}]

        # 2. Create Observation (For Lab Reports)
        # Using LOINC code for Hemoglobin (718-7) per PDF standards
        obs = Observation(
            status="final",
            code={"coding": [{"system": "http://loinc.org", "code": "718-7", "display": extracted["lab_test"]}]},
            subject={"reference": f"Patient/{patient.id}"},
            valueQuantity={"value": extracted["lab_result"], "unit": extracted["unit"]}
        )
        obs.id = str(uuid.uuid4())

        # 3. Create MedicationRequest (For Prescriptions)
        med = MedicationRequest(
            status="active",
            intent="order",
            medicationCodeableConcept={"coding": [{"system": "http://snomed.info/sct", 
                                                  "code": extracted["snomed_code"], 
                                                  "display": extracted["drug"]}]},
            subject={"reference": f"Patient/{patient.id}"}
        )
        med.id = str(uuid.uuid4())

        # --- STAGE 4: WRAP IN BUNDLE ---
        bundle = Bundle(type="collection")
        bundle.entry = [
            BundleEntry(resource=patient),
            BundleEntry(resource=obs),
            BundleEntry(resource=med)
        ]

        return bundle.json(indent=2)

# --- EXECUTION ---
if __name__ == "__main__":
    pipeline = CoreFHIRPipeline()
    # Replace with your medical image file
    try:
        fhir_json = pipeline.extract_and_convert("medical_record.jpg")
        print(fhir_json)
        
        with open("emr_fhir_record.json", "w") as f:
            f.write(fhir_json)
    except Exception as e:
        print(f"Error: {e}. Check if Tesseract and image are present.")