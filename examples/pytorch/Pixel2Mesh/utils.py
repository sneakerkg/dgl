import logging
import os


def create_logger(log_dir, log_level):
    log_file = 'log.txt'
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    if log_level == "info":
        logger.setLevel(logging.INFO)
    elif log_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError("Log level has to be one of info and debug")
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
