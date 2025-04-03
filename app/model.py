# import torch
# from torchvision import transforms
# from PIL import Image
# from app.config import CONFIG
# from app.model_def import CaptchaModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Characters (same as Part 1)
# idx_to_char = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 
#                9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 
#                17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 
#                25: 'Y', 26: 'Z', 27: '0', 28: '1', 29: '2', 30: '3', 31: '4', 32: '5', 
#                33: '6', 34: '7', 35: '8', 36: '9'}

# model = CaptchaModel(num_chars=len(idx_to_char)).to(device)
# model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=device))
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
#     transforms.ToTensor(),
# ])

# def predict_captcha(image: Image.Image):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image)
#     output = torch.argmax(output, dim=2)
#     pred = []
#     prev = -1
#     for p in output[0]:
#         if p.item() != 0 and p.item() != prev:
#             pred.append(idx_to_char.get(p.item(), ""))
#         prev = p.item()
#     return ''.join(pred)


# import torch
# from torchvision import transforms
# from PIL import Image
# from app.config import CONFIG
# from app.model_def import CaptchaModel
# from app.utils import decode_predictions

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load Model
# model = CaptchaModel().to(device)
# model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=device))
# model.eval()

# # Prediction function
# def predict_captcha(image_file):
#     try:
#         image = Image.open(image_file).convert('L')
#         # transform = transforms.Compose([
#         #     transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
#         #     transforms.ToTensor(),
#         # ])
#         transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5,), std=(0.5,))
#         ])
        
#         image = transform(image).unsqueeze(0).to(device)
#         output = model(image)
#         prediction = decode_predictions(output)
#         return {"prediction": prediction[0]}
#     except Exception as e:
#         return {"error": str(e)}


# ========== PREDICT FUNCTION ==========

# def predict_captcha(image_bytes):
#     transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5,), std=(0.5,))
#     ])

#     image = Image.open(image_bytes)
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(image)
#         prediction = decode_prediction(output, idx_to_char)
#     return prediction




import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import json
from app.model_def import CaptchaModel
from app.utils import decode_predictions
from app.config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character mapping (must match your Part 1)
idx_to_char = CONFIG.IDX_TO_CHAR

# # Initialize and load model
# model = CaptchaModel(num_chars=len(idx_to_char))
# model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
#     transforms.ToTensor(),
# ])

# # Load idx_to_char
# with open(CONFIG.IDX_TO_CHAR_PATH, "r") as f:
#     idx_to_char = json.load(f)

# Load model
model = CaptchaModel(num_chars=CONFIG.NUM_CLASSES)
model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transform (same as val_transform in training)
transform = transforms.Compose([
    transforms.Resize((CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH)),
    transforms.ToTensor(),
])

def predict_captcha(pil_image):
    try:
        # Image is already PIL Image, so no need to open again
        image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = decode_predictions(output, idx_to_char)
            print(output)
            print(prediction)
        return prediction[0]
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# def predict_captcha(image):
#     try:
#         # Apply transform
#         image = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(image)
#             prediction = decode_predictions(output)
#             print(f"Raw Output: {output}")
#             print(f"Decoded: {prediction}")
#         return prediction[0]
#     except Exception as e:
#         return f"Error in prediction: {str(e)}"

# def predict_captcha(image_bytes):
#     try:
#         # image = Image.open(image_bytes).convert('L')
#         image = Image.open(BytesIO(image_bytes)).convert('L')
#         image = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(image)
#             prediction = decode_predictions(output, idx_to_char)
#             print(output)
#             print(prediction)
#         return prediction[0]
#     except Exception as e:
#         return f"Error in prediction: {str(e)}"
