import logging, json, os, time
from functools import wraps
from dotenv import load_dotenv

class AppConfig:
    def __init__(self):
        self.ENV = os.getenv("APP_ENV", "development")
        load_dotenv(f"configs/{'prod' if self.ENV == 'production' else 'dev'}.env")
config = AppConfig()

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time()-start:.4f}s")
        return res
    return wrapper
