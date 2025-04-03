import json

class Config:
    MODEL_PATH = "./model/captcha_model_improved.pth"
    IDX_TO_CHAR_PATH = "./model/idx_to_char.json"

    # Model parameters
    IMG_WIDTH = 65
    IMG_HEIGHT = 25
    DROPOUT_RATE = 0.3

    # Load mapping
    with open(IDX_TO_CHAR_PATH, "r") as f:
        IDX_TO_CHAR = json.load(f)
    NUM_CLASSES = len(IDX_TO_CHAR)


CONFIG = Config()
