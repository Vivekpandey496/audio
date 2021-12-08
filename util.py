import logging
import logging.handlers
import os
from functools import wraps
from time import time
from constant import LOGS_PATH


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.handlers.RotatingFileHandler(
        log_file, mode='a', maxBytes=5*1024*1024, backupCount=50, encoding=None, delay=False)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# Access file logger
access_log = setup_logger('first_logger', os.path.join(LOGS_PATH, 'access.log'))
# Error file logger
error_log = setup_logger('second_logger', os.path.join(LOGS_PATH, 'error.log'))


def execution_time(decorated_function):
    """
    Decorator for Calculating time execution of function
    :param decorated_function:
    :return:
    """

    @wraps(decorated_function)
    def wrapper(*args, **kwargs):
        func_name = decorated_function.__name__
        start = time()
        result = decorated_function(*args, **kwargs)
        end = time()
        # Access file logger
        access_log.info("Elapsed time for %s is %s", func_name, end - start)
        return result

    return wrapper
