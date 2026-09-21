"""
logging_config.py
=================
This module contains the logging configuration for the dendritic_modeling
package.
"""

import logging
import os
import sys

LOGGER_NAME = "dendritic_modeling"


class LoggerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = True

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(levelname)s %(filename)s:%(lineno)d  %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # Default log directory - use environment variable or fallback to logs directory
        default_log_dir = os.environ.get("DENDRITIC_LOG_DIR", "logs")
        if not os.path.isabs(default_log_dir):
            # If relative path, make it relative to project root
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            default_log_dir = os.path.join(project_root, default_log_dir)

        os.makedirs(default_log_dir, exist_ok=True)
        self.file_path = os.path.join(default_log_dir, LOGGER_NAME + ".log")
        self._set_file_handler(self.file_path)

    def _set_file_handler(self, file_path):
        self.logger.handlers = [
            h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)
        ]
        try:
            # Ensure the directory exists
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f"Failed to set file handler: {e}")

    def set_log_file(self, file_path):
        self.file_path = file_path
        self._set_file_handler(file_path)
        self.logger.info(f"Log file set to {file_path}.")

    def get_logger(self):
        return self.logger

    def set_log_directory(self, log_dir):
        """Set the directory where log files should be saved."""
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, os.path.basename(self.file_path))
        self.set_log_file(log_file)


# Wrapper functions for backward compatibility
def setup_logging(log_dir=None, debug=False):
    """Setup logging with optional directory and debug level.

    Args:
        log_dir (str, optional): Directory to save log files.
                               If None, uses default from environment or 'logs'
        debug (bool, optional): If True, sets console logging to DEBUG level

    Returns:
        logging.Logger: The configured logger instance
    """
    manager = LoggerManager()

    if log_dir:
        manager.set_log_directory(log_dir)

    if debug:
        # Set console handler to DEBUG level
        for handler in manager.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.DEBUG)
                break

    return manager.get_logger()


def get_logger(name=None):
    """Get the logger instance.

    Args:
        name (str, optional): Logger name (ignored, maintained for compatibility)

    Returns:
        logging.Logger: The configured logger instance
    """
    manager = LoggerManager()
    return manager.get_logger()
