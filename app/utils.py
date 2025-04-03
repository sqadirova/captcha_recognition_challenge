import torch
from app.config import CONFIG
import json

BLANK_TOKEN = 0  

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
            if p_item != BLANK_TOKEN and p_item != prev_char:
                if str(p_item) in idx_to_char:
                    text += idx_to_char[str(p_item)]
            prev_char = p_item
        predictions.append(text)

    return predictions
