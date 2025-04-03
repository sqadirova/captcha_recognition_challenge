import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import json
from app.model_def import CaptchaModel
from app.utils import decode_predictions
from app.config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character mapping
idx_to_char = CONFIG.IDX_TO_CHAR

# Load model
model = CaptchaModel(num_chars=CONFIG.NUM_CLASSES)
model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
    transforms.ToTensor(),
])

def predict_captcha(pil_image):
    try:
        image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probs = torch.exp(output) 
            max_probs, _ = torch.max(probs, dim=2)
            confidence = max_probs.mean().item() * 100

            prediction = decode_predictions(output, idx_to_char)
        return {"text": prediction[0], "confidence": round(confidence, 2)}
    except Exception as e:
        return f"Error in prediction: {str(e)}"

