import os
import sys
import json
import uuid
from datetime import datetime
from groq import Groq

# --- CONFIGURATION ---
# Paste your Groq API Key here
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"

class GroqEMREngine:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        # We use Llama 3.3 70B for its high clinical reasoning accuracy
        self.model = "llama-3.3-70b-versatile"

    def process_ocr(self, raw_text):
        prompt = f"""
        Act as a Medical Information Extraction System (Stage 4 & 5 of EMR Pipeline).
        Convert the following noisy OCR text into a structured HL7 FHIR Observation Bundle.

        REQUIRED TASKS:
        1. OCR HEALING: Correct medical misspellings (e.g., 'Haemoglobln' -> 'Hemoglobin').
        2. ENTITY EXTRACTION: Find every test result and unit.
        3. SHIELDING: Isolate the 'Patient Result' from the 'Reference Range'. Ignore flags like [H] or [L].
        4. STANDARDIZATION: Map each test to its standard LOINC code.
        5. OUTPUT: Return ONLY a valid JSON FHIR Bundle.

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