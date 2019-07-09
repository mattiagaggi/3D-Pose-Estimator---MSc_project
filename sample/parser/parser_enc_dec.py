from sample.base.base_parser import BaseParser



class EncParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.001)
        self._add_batch_size(64)
        self._add_epochs(100)
        self._add_name('encoder_decoder')
        self._add_resume(False)
        self._add_input_path()
        self._add_output_dir("sample/checkpoints")
        self._add_model_checkpoints(5000)
        self._add_verbose(10, 50, 1000, 200)
        self._add_data_threads(8)
        self._add_cuda()
        self._add_reset()
