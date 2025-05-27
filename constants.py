try:
    from config import *
except ImportError:
    pass

INPUT_SHAPE = (128, 128, 3)
LABEL_TO_INDEX = {
    "happy": 0,
    "sad": 1,
    "angry": 2
}

MODEL_FILENAME = "model.keras"
