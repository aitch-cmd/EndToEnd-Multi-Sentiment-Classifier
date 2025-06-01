import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # Set both handlers to DEBUG level to see debug messages
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG) # Changed from INFO to DEBUG

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG) # Changed from INFO to DEBUG

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Configure the logger when module is imported
configure_logger()


# Logger for S3 

# def configure_logger():
#     """
#     Configures logging with proper filtering for third-party libraries
#     """
#     logger = logging.getLogger()
    
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # Key change: Set root logger to INFO by default
#     logger.setLevel(logging.INFO)

#     formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

#     # File handler: INFO level for general tracking
#     file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, 
#                                      backupCount=BACKUP_COUNT, encoding="utf-8")
#     file_handler.setFormatter(formatter)
#     file_handler.setLevel(logging.INFO)

#     # Console handler: DEBUG level for detailed output
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(formatter)
#     console_handler.setLevel(logging.DEBUG)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     # Critical: Silence noisy libraries
#     noisy_libraries = [
#         'botocore', 'boto3', 'urllib3', 
#         's3transfer', 'matplotlib', 'fsspec'
#     ]
#     for lib in noisy_libraries:
#         logging.getLogger(lib).setLevel(logging.WARNING)

#     return logger

# # Configure the logger when module is imported
# configure_logger()