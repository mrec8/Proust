"""
Logging utilities for the narrative agent.
"""
import os
import logging

class NarrativeAgent:
    # ...existing code...
    def setup_logging(self, log_dir: str, log_level: str = "INFO") -> None:
        """
        Configures logging for the narrative agent.
        
        Args:
            log_dir: Directory to save log files
            log_level: Logging level (e.g., "INFO", "DEBUG")
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure the logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create a file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "agent.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Add handlers to the root logger
        logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
    # ...existing code...