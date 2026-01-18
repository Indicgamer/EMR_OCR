import os
from google.colab import userdata
import subprocess
import time

# 1. Set the Secret Key
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

# 2. Nuclear Cleanup
!pkill streamlit
!pkill cloudflared

# 3. Start Streamlit (Background)
# We give it 10 seconds to load the heavy PaddleOCR model into cache
print("Starting EMR Server...")
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(10)

# 4. Start Tunnel
print("\n" + "="*50)
print("CLICK THE .trycloudflare.com LINK BELOW")
print("="*50 + "\n")
!cloudflared tunnel --url http://127.0.0.1:8501