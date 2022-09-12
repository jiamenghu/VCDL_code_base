import os
import time
import os.path as op


def mkdir(dir_path):
    """Create directory.
    
    Args:
        dir_path (str)
    """
    assert not op.exists(dir_path), ("Dir already exists!")
    os.makedirs(dir_path)

