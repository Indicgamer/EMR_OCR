# 1. Set environment variables for stability
import os
os.environ['PIP_DEFAULT_TIMEOUT'] = '1000'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

print("--- 1. Purging incompatible libraries ---")
!pip uninstall -y numpy paddleocr paddlepaddle-gpu paddlex langchain langchain-community opencv-python-headless -q

print("--- 2. Installing Stable GPU Stack (Paddle 2.6.1) ---")
# Locking paddlepaddle-gpu to 2.6.1 avoids the 'set_optimization_level' bug
!pip install paddlepaddle-gpu==2.6.1 -q

print("--- 3. Installing PaddleOCR & NLP Tools ---")
!pip install paddleocr==2.7.3 groq thefuzz python-Levenshtein -q

print("--- 4. Installing Langchain & OpenCV Fixes ---")
# These specific versions fix the 'ModuleNotFoundError: langchain.docstore'
!pip install langchain==0.1.0 langchain-community==0.0.34 --no-deps -q
!pip install opencv-python-headless==4.9.0.80 -q

print("--- 5. LOCKING NUMPY TO 1.26.4 (Critical) ---")
# PaddlePaddle 2.6.1 will CRASH on Numpy 2.0+. We must force 1.26.4.
!pip install "numpy<2.0,>=1.26.0" --force-reinstall --no-deps -q

print("\n" + "="*60)
print("SETUP COMPLETE. Session will now RESTART.")
print("Wait 10 seconds, then run the MedicalLayoutOCR code.")
print("="*60 + "\n")

import os
os.kill(os.getpid(), 9)