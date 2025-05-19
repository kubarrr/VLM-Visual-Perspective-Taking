import os
import logging
from logging import Logger

def setup_logger(name, log_file) -> Logger:
    """Set up a logger for a specific run."""
    
    # make sure folder exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # set logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if logger is reused
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger