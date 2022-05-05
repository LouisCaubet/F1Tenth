import json
import os

_is_loaded = False


def load_config():

    global _is_loaded
    if _is_loaded:
        return
    _is_loaded = True

    # Load config
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config.json"))
    CONFIG = json.load(open(cfg_path, "r", encoding="utf8"))

    # Define config options
    global DATASET_FOLDER, DEBUG_GENERATE_ONLY_ONE_DATASET, DATASET_LIST
    global DEFAULT_PTH_FILE, GYM_MAP_NAME, RENDER
    global SB3_MODEL_TO_USE, GYM_USE_RL_MODEL, RL_RESUME_TRAINING
    global DATASET_FILTER_ONLY_TURNS, RESUME_TRAINING, PTH_PP_MODEL

    DATASET_FOLDER = CONFIG['DATASET_FOLDER']
    DEBUG_GENERATE_ONLY_ONE_DATASET = CONFIG['DEBUG_GENERATE_ONLY_ONE_DATASET']
    DATASET_LIST = CONFIG['DATASETS']
    DEFAULT_PTH_FILE = CONFIG["PTH_MODEL_TO_USE"]
    GYM_MAP_NAME = CONFIG["GYM_MAP"]
    RENDER = CONFIG["GYM_ENABLE_RENDER"]
    SB3_MODEL_TO_USE = CONFIG["SB3_MODEL_TO_USE"]
    GYM_USE_RL_MODEL = CONFIG["GYM_USE_RL_MODEL"]
    RL_RESUME_TRAINING = CONFIG["RL_RESUME_TRAINING"]
    DATASET_FILTER_ONLY_TURNS = CONFIG["DATASET_FILTER_ONLY_TURNS"]
    RESUME_TRAINING = CONFIG["RESUME_TRAINING"]
    PTH_PP_MODEL = CONFIG["PTH_PP_MODEL"]


load_config()
