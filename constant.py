import os


ABSOLUTE_MODEL_PATH = '/home/sumit/ml_models/'
HASH_MAP_PATH = os.path.join(os.getcwd(), 'static')
LOGS_PATH = os.path.join(os.getcwd(), 'logs')
MODEL_PATH = os.path.join(ABSOLUTE_MODEL_PATH)
CLASS_NAMES = ["negative", "neutral", "positive"]
LANG_CLASS_NAMES = ["en", "hi"]
REMOVAL_WORDS = ["mere", "behind", "black", "miss", "cow", "time everyday"]
AUDIO_DIR = 'media/audio'
