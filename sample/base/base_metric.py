from torch.nn import Module
from logger.console_logger import ConsoleLogger

class BaseMetric(Module):
    """Base Metric class"""

    def __init__(self):
        """Initialize class"""
        super().__init__()
        self.name = self.__class__.__name__
        self._logger = ConsoleLogger(self.name)

    def forward(self, *input):


        raise NotImplementedError('Abstract method in BaseMetric class...')


    def log_model(self, logger, iteration, error, added_name=None):
        if added_name is None:
            logger.add_scalar('metrics/{0}'.format(self.name),
                            error,
                            iteration)
        else:
            logger.add_scalar(added_name,
                              error,
                              iteration)


    def log_train(self,logger, iteration, error,train=False, added_name = None):
        """Add result to log file

        Arguments:
            logger {Logger} -- class responsible for logging into
            iteration {int} -- iteration number
            error {float} -- error value
        """
        if train:
            if added_name is None:
                logger.record_scalar('train_metrics/{0}'.format(self.name),
                                 error,
                                iteration)
            else:
                logger.record_scalar('train_'+added_name,
                                     error,
                                     iteration)
        else:
            if added_name is None:
                logger.record_scalar('test_metrics/{0}'.format(self.name),
                                     error,
                                     iteration)
            else:
                logger.record_scalar('test_' + added_name,
                                     error,
                                     iteration)


