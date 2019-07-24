
import numpy as np
import os


from utils.io import ensure_dir

class TrainingLogger:
    """
    Logger, used by save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name
        path=os.path.join(self.dir_path, 'training_files/')
        ensure_dir(path)
        self.losses={'training_loss':[]),
                    'test_loss':[]}

    def save_loss(self):
        self.lo