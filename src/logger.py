import logging
import sys
from pathlib import Path

# Fix for MacOS OpenMP runtime error (needed if run outside the Docker container)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_logger(name: str) -> logging.Logger:
    """
    Creates a logger that writes to both the console and the persistent log file.
    """
    
    # Define Paths (Two levels up from src/utils/ to reach project root)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    LOG_DIR = BASE_DIR / "reports"
    LOG_FILE = LOG_DIR / "system_errors.log"

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(str(LOG_FILE), encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger