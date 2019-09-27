from sample.base.base_parser import BaseParser



class GAN_Parser(BaseParser):
    """Train parser"""
    def __init__(self, description):
        super().__init__(description)
        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(100)
        self._add_name("gan")
        self._add_output_dir("data/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_data_threads(0)
        self._add_verbose(50, 30, 500, 100) #verb iter, train_log_step,img log_step
