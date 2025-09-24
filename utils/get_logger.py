import logging
import os
import sys

def get_logger(save_dir, distributed_rank, filename="test_new.log", stdout=False, mode='a'):
    """
    Get the logger.
    """
    if distributed_rank > 0: 
        logger_not_root = logging.getLogger(name=__name__) 
        logger_not_root.propagate = False
        return logger_not_root

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    
    if stdout:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)
 
    if save_dir is not None:
        save_file = os.path.join(save_dir, filename)
        fh = logging.FileHandler(save_file, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
     
    return root_logger
