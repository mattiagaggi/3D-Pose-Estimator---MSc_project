
import argparse
from sample.base.base_logger import FrameworkClass
from logger.console_logger import ConsoleLogger


class BaseParser(FrameworkClass):
    """Base parser class"""

    def __init__(self, description):
        """Initialization"""
        super().__init__()

        self.parser = argparse.ArgumentParser(description=description)
        self._logger=ConsoleLogger("Parser")


    def _add_batch_size(self, default):
        """Add batch-size argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-b',
            '--batch-size',
            default=default,
            type=int,
            help='mini-batch size (default: {:d})'.format(default))

    def _add_name(self, default):
        """Add name argument

        Arguments:
            default {str} -- default value
        """

        assert isinstance(default, str)

        self.parser.add_argument(
            '-n',
            '--name',
            default=default,
            type=str,
            help='output name (default: {})'.format(default))

    def _add_output_dir(self, default):
        """Add oytput directory argument

        Arguments:
            default {str} -- default value
        """

        assert isinstance(default, str)

        self.parser.add_argument(
            '-o',
            '--output',
            default=default,
            type=str,
            help='output directory path (default: {})'.format(default))

    def _add_input_path(self):
        """Add input argument"""

        self.parser.add_argument(
            '-i',
            '--input',
            required=True,
            type=str,
            help='input directory/file path')

    def _add_resume(self, required, short='-r', long='--resume', default='init'):
        """Add resume argument

        Arguments:
            required {bool} -- is this required
            short {str} -- format
            long {str} -- format
        """

        assert isinstance(required, bool)

        self.parser.add_argument(
            short,
            long,
            default=default,
            required=required,
            type=str,
            help='input directory/file path containing the model')

    def _add_data_threads(self, default):
        """Add num threahds argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-t',
            '--num-threads',
            default=default,
            type=int,
            help='number of threads for dataset loader (default: {:d})'.format(
                default))



    def _add_learning_rate(self, default):
        """Add learning rate argument

        Arguments:
            default {float} -- default value
        """

        assert isinstance(default, float)

        self.parser.add_argument(
            '-lr',
            '--learning-rate',
            default=default,
            type=float,
            help='learning rate (default: {:.6f})'.format(default))

    def _add_regularizer_weight(self, name='--reg-weight', default=0.0):
        """Add learning rate argument

        Arguments:
            default {float} -- default value
        """

        assert isinstance(default, float)

        self.parser.add_argument(
            name,
            default=default,
            type=float,
            help='regularizer weight (default: {:.6f})'.format(default))

    def _add_epochs(self, default):
        """Add epochs argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-e',
            '--epochs',
            default=default,
            type=int,
            help='number of epochs (default: {:d})'.format(default))





    def _add_model_checkpoints(self, iterations):
        """Add argument to save model every n iterations

        Arguments:
            iterations {int} -- number of iterations
        """

        self.parser.add_argument(
            '--save-freq',
            default=iterations,
            type=int,
            help='training checkpoint frequency in iterations(default: {:d})'.format(iterations))

    def _add_verbose(self, text, train,img_log_step, test_log_step):
        """Add arguments for verbose

        Arguments:
            text {int} -- number of iterations
            train {int} -- number of iterations
            image {int} -- number of iterations

        Keyword Arguments:
            single_epoch {bool} -- code running for a single epoch (default: {False})
        """

        self.parser.add_argument(
            '-v',
            '--verbosity',
            default=2,
            type=int,
            help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
        self.parser.add_argument(
            '--verbosity-iter',
            default=text,
            type=int,
            help='vlog frequency for text (default: {:d})'.format(text))
        if train > 0:
            self.parser.add_argument(
                '--train-log-step',
                default=train,
                type=int,
                help='log frequency for training (default: {:d})'.format(train))
        if img_log_step > 0:
            self.parser.add_argument(
                '--img-log-step',
                default=img_log_step,
                type=int,
                help='log frequency for image (default: {:d})'.format(img_log_step))
        if test_log_step > 0:
            self.parser.add_argument(
                '--test-log-step',
                default=test_log_step,
                type=int,
                help='log frequency for training (default: {:d})'.format(train))
        if img_log_step % train !=0:
            self._logger.error("train images might be never recorder")
        if img_log_step % test_log_step !=0:
            self._logger.error("test images might be never recorder")

    def get_arguments(self):
        """Get arguments"""

        return self.parser.parse_args()
