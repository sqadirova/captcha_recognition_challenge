from starlette.config import Config as StarletteConfig
import json

# Read .env
env = StarletteConfig(".env")

class Config:
    MODEL_PATH = env("MODEL_PATH", cast=str)
    IDX_TO_CHAR_PATH = env("IDX_TO_CHAR_PATH", cast=str)
    IMG_WIDTH = env("IMG_WIDTH", cast=int)
    IMG_HEIGHT = env("IMG_HEIGHT", cast=int)
    DROPOUT_RATE = env("DROPOUT_RATE", cast=float)

    # Load mapping
    with open(IDX_TO_CHAR_PATH, "r") as f:
        IDX_TO_CHAR = json.load(f)
    NUM_CLASSES = len(IDX_TO_CHAR)


CONFIG = Config()
