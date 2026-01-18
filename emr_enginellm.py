import os
import sys
import json
import uuid
from datetime import datetime
from groq import Groq
from google.colab import userdata

# --- CONFIGURATION ---
# Paste your Groq API Key here
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

class GroqEMREngine:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

    def process_ocr(self, raw_text):
        prompt = f"""
        Act as a Medical Information Extraction System (Stage 4 & 5 of EMR Pipeline).
        Convert the following noisy OCR text into a structured HL7 FHIR Observation Bundle.

        REQUIRED TASKS:
        1. PATIENT/DOCTOR IDENTIFICATION: Extract Patient Name, Age, Sex, and Doctor Name.
        2. OCR HEALING: Correct medical misspellings (e.g., 'Haemoglobln' -> 'Hemoglobin').
        3. ENTITY EXTRACTION: Extract ALL medical tests, results, and units.
        4. SHIELDING: Isolate the 'Patient Result' from the 'Reference Range'. Ignore flags like [H] or [L].
        5. STANDARDIZATION: Map each test to its standard LOINC code.
        6. OUTPUT: Return ONLY a valid JSON FHIR Bundle.

        OCR TEXT:
        {raw_text}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical data parser that outputs only valid JSON FHIR Bundles."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=self.model,
                # Forces the model to output a valid JSON object
                response_format={"type": "json_object"},
                temperature=0.1, 
            )
            
            return json.loads(chat_completion.choices[0].message.content)
            
        except Exception as e:
            return {
                "resourceType": "Bundle",
                "error": str(e),
                "entry": []
            }

if __name__ == "__main__":
    # Get piped text from PaddleOCR (medical_ocr.py)
    raw_ocr = sys.stdin.read()
    
    if not raw_ocr.strip():
        print(json.dumps({"error": "No OCR input found"}))
        sys.exit(1)

    engine = GroqEMREngine()
    result = engine.process_ocr(raw_ocr)
    
    # Final Structured Output
    print(json.dumps(result, indent=2))