import logging
import sys
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    """
    Creates a logger that writes to both:
    1. The console (Terminal)
    2. A persistent file (reports/system_errors.log)
    """
    
    # 1. Define Paths
    # Current file: src/utils/logger.py
    # Root dir:     src/utils/../../  (Two levels up)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    LOG_DIR = BASE_DIR / "reports"
    LOG_FILE = LOG_DIR / "system_errors.log"

    # 2. Create 'reports' directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Configure Format
    # Format: [Time] - [Module Name] - [Level] - [Message]
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. Setup Handlers
    
    # File Handler (Saves logs to disk)
    file_handler = logging.FileHandler(str(LOG_FILE), encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Stream Handler (Shows logs in terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 5. Initialize Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs if function is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Self-test block: Run this file directly to check if the log file is created
if __name__ == "__main__":
    test_logger = setup_logger("TestLogger")
    test_logger.info("✅ Logging system is working.")
    test_logger.info(f"📁 Log file should be at: {Path(__file__).resolve().parent.parent.parent / 'reports/system_errors.log'}")