import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    # fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # color_fmt = colored('[%(asctime)s %(name)s]', 'blue') + \
    #             colored('(%(filename)s %(lineno)d)', 'green') + colored(': %(levelname)s %(message)s', 'red')
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'blue') + \
                colored('(%(filename)s %(lineno)d)', 'green') + colored(': %(message)s', 'red')

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger