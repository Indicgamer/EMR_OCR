Electronic Medical Record (EMR) System
A Standardized FHIR-Compliant Pipeline for Automated Clinical Digitization
![alt text](https://img.shields.io/badge/Standards-HL7%20FHIR%20R4-blue)

![alt text](https://img.shields.io/badge/Ontology-LOINC-green)

![alt text](https://img.shields.io/badge/Framework-ABDM%20Ready-orange)
📖 Project Overview
The digitization of physical medical records (lab reports, prescriptions, discharge summaries) remains a critical bottleneck in achieving universal healthcare interoperability. This project presents a State-of-the-Art (SOTA) Clinical Pipeline that automates the transition from unstructured physical documents to structured, machine-readable HL7 FHIR R4 resources.
The system utilizes a hybrid architecture of Layout-Aware Computer Vision and Semantic Named Entity Recognition (NER) to ensure high-fidelity extraction even from noisy, low-resolution medical scans.
🛠 Clinical Pipeline Architecture
The system implementation is divided into four distinct clinical stages as defined in our research methodology:
1. Stage 1: Document Layout Analysis (Vision)
Utilizing PaddleOCR with a vertical-alignment heuristic, the system reconstructs the spatial relationship between clinical parameters and numeric results.
Feature: Groups fragmented text into horizontal "Clinical Rows."
Benefit: Prevents the "column-mixing" error common in standard OCR engines.
2. Stage 2: Named Entity Recognition (NER) & NLP
A specialized NLP Engine performs deep semantic analysis on the raw text to identify:
Patient Metadata: Name, Age, Gender, and Local IDs.
Clinical Entities: Test descriptions, numeric results, and units of measure.
Normalization: Automated healing of OCR character distortions (e.g., "H3moglob1n" ➔ "Hemoglobin").
3. Stage 3: Semantic Normalization (Ontology Mapping)
Extracted entities are mapped to the LOINC (Logical Observation Identifiers Names and Codes) system.
Interoperability: Maps local test names (e.g., "HB") to standard codes (e.g., 718-7).
Standardization: Ensures the record is compliant with the National Digital Health Stack.
4. Stage 4: HL7 FHIR R4 Generation
The final output is encapsulated in an HL7 FHIR Observation Bundle, containing:
Patient Resource: Demographics.
Practitioner Resource: Ordering Physician details.
Observation Resource: Coded laboratory findings with status and category metadata.
✨ Key Technical Features
Fuzzy Semantic Triggering: Identifies clinical tests even with significant spelling mismatches using Levenshtein-based scoring.
Reference Range Shielding: Logically distinguishes between "Patient Values" and "Biological Reference Ranges" using contextual proximity logic.
Clinical Validation Dashboard: A professional interface for "Human-in-the-Loop" (HITL) verification, as required by clinical safety standards.
📂 Repository Structure
File	Description
app.py	Integrated Streamlit Portal (Visualization & Validation UI)
medical_ocr.py	Vision Engine for Layout-Aware Text Extraction
emr_enginellm.py	Semantic NER & FHIR Mapping Engine
data/	Dataset containing Lab Reports and Prescription samples
README.md	System Documentation
🚥 Deployment & Execution (Google Colab)
1. Environment Initialization
Run the "One-Shot Setup" provided in the notebook to install the required GPU stack:
PaddlePaddle GPU v2.6.1
Numpy v1.26.4 (Locked for Python 3.12 compatibility)
2. Configure Credentials
Store your specialized Clinical NER API key in the Colab Secrets menu:
Name: GROQ_API_KEY
3. Launch System
code
Bash
# Pull latest updates
git pull

# Execute Integrated Pipeline
python /content/EMR_OCR/app.py
📋 Compliance & Standards
This system is designed to adhere to the following medical and legal frameworks:
Interoperability: HL7 FHIR R4 (JSON Schema).
Ontology: LOINC (Laboratory) / SNOMED CT (Clinical Findings).
Data Governance: IT Act 2000 & DISHA Bill (Proposed) for medical data localization.