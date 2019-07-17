from sample.base.base_parser import BaseParser



class EncParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(30)
        self._add_name('enc_dec_can')
        self._add_output_dir("sample/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_verbose(50, 10, 300) #verb iter, train_log_step,img log_step
        self._add_data_threads(2)

