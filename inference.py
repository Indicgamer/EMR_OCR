import torch
from PIL import Image
from model import create_model

class EMRPredictor:
    def __init__(self, model_path):
        self.model, self.processor = create_model()
        # Load your trained weights
        state_dict = torch.load(f"{model_path}/pytorch_model.bin")
        self.model.load_state_dict(state_dict)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]