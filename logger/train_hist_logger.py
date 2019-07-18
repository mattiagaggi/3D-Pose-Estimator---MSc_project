
from utils.io import ensure_dir
import numpy as np

class TrainingLogger:
    """
    Logger, used by save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name
        path=os.path.join(self.dir_path, 'training_pickled/')
        ensure_dir(path)
        self.losses={'training_loss':np.array([]),
                    'test_loss':np.array([])}

    def