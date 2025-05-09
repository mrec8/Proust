"""
Logging utilities for the narrative agent.
"""
import os
import logging
from typing import Optional

def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """
    Configures logging for the narrative agent.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., "INFO", "DEBUG")
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert string log level to Python logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure the logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "agent.log"))
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging configured at level {log_level} in directory {log_dir}")