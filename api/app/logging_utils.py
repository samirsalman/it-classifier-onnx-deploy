import logging


def setup_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    file_handler_info = logging.FileHandler(filename="logging/info.txt")
    file_handler_info.setLevel(logging.INFO)
    file_handler_error = logging.FileHandler(filename="logging/error.txt")
    file_handler_error.setLevel(logging.ERROR)
    logging_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler_info.setFormatter(logging_format)
    file_handler_error.setFormatter(logging_format)
    logger.addHandler(file_handler_info)
    logger.addHandler(file_handler_error)
    return logger
