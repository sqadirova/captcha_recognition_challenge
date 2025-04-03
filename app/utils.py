import torch
# from torchvision import transforms
# from PIL import Image
from app.config import CONFIG
import json


BLANK_TOKEN = 0  

# def decode_predictions(outputs, idx_to_char):
#     """Convert model outputs to text predictions"""
#     predictions = []
#     output_args = torch.argmax(outputs.detach().cpu(), dim=2)

#     for pred in output_args:
#         text = ''
#         prev_char = None
#         for p in pred:
#             p_item = p.item()
#             if p_item != BLANK_TOKEN and p_item != prev_char:
#                 if p_item in idx_to_char:
#                     text += idx_to_char[p_item]
#             prev_char = p_item
#         predictions.append(text)
#     return predictions

# Load idx_to_char
with open(CONFIG.IDX_TO_CHAR_PATH, "r") as f:
    idx_to_char = json.load(f)

def decode_predictions(outputs, idx_to_char):
    """Convert model outputs to text predictions"""
    predictions = []
    output_args = torch.argmax(outputs.detach().cpu(), dim=2)

    for pred in output_args:
        text = ''
        prev_char = None
        for p in pred:
            p_item = p.item()
            if p_item != 0 and p_item != prev_char:  # 0 is BLANK_TOKEN
                if str(p_item) in idx_to_char:
                    text += idx_to_char[str(p_item)]
            prev_char = p_item
        predictions.append(text)

    return predictions


# def decode_predictions(outputs):
#     """Decode model output to text"""
#     predictions = []
#     output_args = torch.argmax(outputs.detach().cpu(), dim=2)

#     for pred in output_args:
#         text = ""
#         prev_char = None
#         for p in pred:
#             p_item = p.item()
#             if p_item != 0 and p_item != prev_char:
#                 text += idx_to_char.get(str(p_item), "")
#             prev_char = p_item
#         predictions.append(text)
#     return predictions

