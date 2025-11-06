import logging
import os


def get_logger(name: str, log_file: str = None, file_mode: str = "a", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    # Ensure we don't add duplicate handlers on repeated calls
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add or reuse a FileHandler if a log_file is provided
    if log_file:
        # ensure directory exists
        dirpath = os.path.dirname(log_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # check for existing file handler for the same file
        file_abs = os.path.abspath(log_file)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == file_abs
            for h in logger.handlers
        )

        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, mode=file_mode)
            file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger