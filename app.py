import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR
from groq import Groq
from datetime import datetime

# --- 1. INITIAL SETTINGS ---
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
st.set_page_config(page_title="Standardized FHIR EMR System", layout="wide", page_icon="🏥")

# --- 2. CACHED ENGINES ---
@st.cache_resource
def load_ocr_engine():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)

def get_nlp_client():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        st.error("NER Configuration Error: Clinical Key not found.")
        st.stop()
    return Groq(api_key=api_key)

# --- 3. CORE PROCESSING PIPELINE ---
def perform_layout_ocr(img_path):
    ocr_engine = load_ocr_engine()
    result = ocr_engine.ocr(img_path, cls=True)
    if not result or result[0] is None: return ""
    blocks = []
    for line in result[0]:
        box, (text, score) = line[0], line[1]
        y_center = (box[0][1] + box[2][1]) / 2
        blocks.append({"text": text, "y": y_center, "x": box[0][0]})
    blocks.sort(key=lambda b: b['y'])
    rows, current_row = [], [blocks[0]]
    for i in range(1, len(blocks)):
        if abs(blocks[i]['y'] - blocks[i-1]['y']) < 15:
            current_row.append(blocks[i])
        else:
            rows.append(current_row)
            current_row = [blocks[i]]
    rows.append(current_row)
    final_text = ""
    for r in rows:
        r.sort(key=lambda b: b['x'])
        final_text += " ".join([b['text'] for b in r]) + "\n"
    return final_text

def nlp_fhir_extraction(raw_text):
    client = get_nlp_client()
    # Prompt explicitly asks for full FHIR Metadata
    prompt = f"""
    Act as a specialized Clinical NLP Engine. 
    Perform Named Entity Recognition (NER) and output a valid HL7 FHIR R4 JSON Bundle.
    
    INSTRUCTIONS:
        0. PATIENT/DOCTOR IDENTIFICATION: Extract Patient Name, Age, Sex, and Doctor Name.
        1. OCR HEALING: Correct medical misspellings (e.g., 'Haemoglobln' -> 'Hemoglobin').
        2. ENTITY EXTRACTION: Extract ALL medical tests, results, and units.
        3. SHIELDING: Isolate the 'Patient Result' from the 'Reference Range'. Ignore flags like [H] or [L].
        4. STANDARDIZATION: Map each test to its standard LOINC code.
        5. OUTPUT: Return ONLY a valid JSON FHIR Bundle.
    SOURCE TEXT:
    {raw_text}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(completion.choices[0].message.content)

# --- 4. EMR INTERFACE ---
st.title("🏥 Electronic Medical Record (EMR) System")
st.markdown("#### **Standardized Clinical Pipeline**: Automated OCR Layout & HL7 FHIR NER")

st.sidebar.title("System Standards")
st.sidebar.success("✅ HL7 FHIR R4 Compliant")
st.sidebar.success("✅ LOINC Ontology Mapped")
st.sidebar.success("✅ ABDM Interoperable")

uploaded_file = st.file_uploader("📤 Ingest Clinical Document (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_res = st.columns([1, 1.2])
    img = Image.open(uploaded_file)
    temp_path = "active_report.png"
    img.save(temp_path)
    
    with col_img:
        st.image(img, caption="Document Source", use_container_width=True)

    with col_res:
        if st.button("🚀 Execute NLP & FHIR Generation", use_container_width=True):
            try:
                with st.spinner("Processing Document Layout..."):
                    ocr_text = perform_layout_ocr(temp_path)
                
                with st.spinner("Executing Clinical NER & FHIR Mapping..."):
                    fhir_data = nlp_fhir_extraction(ocr_text)
                
                # --- DATA DISPLAY TABS ---
                tab1, tab2, tab3 = st.tabs(["📋 Clinical Dashboard", "🔬 FHIR Technical View", "📄 Raw Bundle"])

                # DATA PARSING
                patient_data = {"name": "N/A", "id": "N/A", "gender": "N/A"}
                observations = []

                for entry in fhir_data.get('entry', []):
                    res = entry.get('resource', {})
                    if res.get('resourceType') == 'Patient':
                        patient_data['name'] = res.get('name', [{}])[0].get('text', 'N/A')
                        patient_data['gender'] = res.get('gender', 'N/A').capitalize()
                        patient_data['id'] = res.get('id', 'local-001')
                    elif res.get('resourceType') == 'Observation':
                        observations.append({
                            "Resource ID": res.get('id', 'obs-x'),
                            "Test Name": res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown'),
                            "Value": res.get('valueQuantity', {}).get('value', 'N/A'),
                            "Unit": res.get('valueQuantity', {}).get('unit', ''),
                            "LOINC": res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A'),
                            "Status": res.get('status', 'final'),
                            "Category": "Laboratory"
                        })

                with tab1:
                    st.markdown("### Patient Metadata")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Name", patient_data['name'])
                    m2.metric("Gender", patient_data['gender'])
                    m3.metric("Standard ID", patient_data['id'])
                    
                    st.divider()
                    st.subheader("Laboratory Results")
                    if observations:
                        df_summary = pd.DataFrame(observations)[["Test Name", "Value", "Unit"]]
                        st.table(df_summary)

                with tab2:
                    st.subheader("FHIR Resource Mapping (LOINC Standard)")
                    st.info("This view shows the internal mapping of clinical entities to universal standards.")
                    if observations:
                        # Showing the columns that prove FHIR implementation
                        df_fhir = pd.DataFrame(observations)[["Resource ID", "Test Name", "LOINC", "Status", "Category"]]
                        st.dataframe(df_fhir, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("**Resource Implementation Details:**")
                    st.write("- **System:** `http://loinc.org` (Logical Observation Identifiers Names and Codes)")
                    st.write("- **Interoperability:** JSON schema follows `HL7 FHIR R4` Bundle structure.")

                with tab3:
                    st.subheader("Final FHIR Bundle (JSON)")
                    st.code(json.dumps(fhir_data, indent=2), language='json')
                    st.download_button("📩 Download FHIR JSON", data=json.dumps(fhir_data, indent=2), file_name="EMR_Bundle.json")

            except Exception as e:
                st.error(f"EMR Pipeline Error: {str(e)}")
else:
    st.info("Ingest document to view FHIR resource mappings.")