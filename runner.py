import os
from google.colab import userdata

# 1. Fetch the key from the 🔑 Secrets menu
# IMPORTANT: Make sure the "Notebook access" toggle is ON for GROQ_API_KEY
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

# 2. Run the pipeline (The '!' command will now see the environment variable)
!python /content/EMR_OCR/medical_ocr.py "your_image_path.png" | python /content/EMR_OCR/emr_enginellm.py
