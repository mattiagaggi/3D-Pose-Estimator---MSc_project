from sample.base.base_parser import BaseParser


name='enc_dec_S15678_rot'
name1=name+"3D"
name2=name+"SMPL2"



class EncParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(5)
        self._add_name(name)
        self._add_output_dir("data/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_verbose(50, 10, 1000, 100) #verb iter, train_log_step,img log_step, test_log_step
        #self._add_data_threads(4)


class Pose_Parser(BaseParser):
    """Train parser"""
    def __init__(self, description):
        super().__init__(description)
        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(5)
        self._add_name(name1)
        self._add_output_dir("data/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_verbose(50, 10, 500, 100) #verb iter, train_log_step,img log_step



class SMPL_Parser(BaseParser):
    """Train parser"""
    def __init__(self, description):
        super().__init__(description)
        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(5)
        self._add_name(name2)
        self._add_output_dir("data/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_verbose(50, 5, 5, 100) #verb iter, train_log_step,img log_step



