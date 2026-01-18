import os
import numpy as np
import paddle
from paddleocr import PaddleOCR

# Force the bypass again for this cell
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

print(f"--- Environment Check ---")
print(f"✅ Numpy Version: {np.__version__}") # Should be 1.26.4
print(f"✅ Paddle Version: {paddle.__version__}") # Should be 2.6.1

try:
    # This will download the models (only happens once)
    engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
    
    # Test on a blank image to confirm it's working
    dummy_img = np.ones((100, 100, 3), dtype='uint8') * 255
    _ = engine.ocr(dummy_img, cls=True)
    
    print("\n🚀 SUCCESS: PaddleOCR is fully functional and ready for your medical reports!")
except Exception as e:
    print(f"\n❌ Error during initialization: {e}")