from torch import nn.Module

class BaseMetric(nn.Module):
    """Base Metric class"""

    def __init__(self):
        """Initialize class"""
        super().__init__()
        self.name= self.__class__.__name__

    def forward(self, *input):


        raise NotImplementedError('Abstract method in BaseMetric class...')


    def log(self, logger, iteration, pred, gt):


        error = self.forward(pred, gt)
        self.log_res(logger,
                     iteration,
                     error)

    def log_model(self, logger, iteration, error):

        logger.add_scalar('metrics/{0}'.format(self.name),
                          error,
                          iteration)

    def log_train(self,logger, iteration, error,train=True):
        """Add result to log file

        Arguments:
            logger {Logger} -- class responsible for logging into
            iteration {int} -- iteration number
            error {float} -- error value
        """
        if train:
            logger.record_scalar('train_metrics/{0}'.format(self.name),
                             error,
                            iteration)
        else:
            logger.record_scalar('test_metrics/{0}'.format(self.name),
                                 error,
                                 iteration)



