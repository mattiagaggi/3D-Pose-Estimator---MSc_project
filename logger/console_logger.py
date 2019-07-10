
import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter"""


    def __init__(self):
        """initializer"""

        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"

        self.DATE = '\033[94m'
        self.GREEN = '\033[92m'
        self.WHITE = '\033[0m'
        self.WARNING = '\033[93m'
        self.RED = '\033[91m'

        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        """format message"""

        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN

        if record.levelno == logging.WARN:
            color = self.WARNING

        if record.levelno == logging.ERROR:
            color = self.RED

        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} %(name)s{}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)

        return logging.Formatter.format(self, record)


class ConsoleLogger():
    """Console logger"""

    def __init__(self, name='main'):
        super().__init__()

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        if not (len(self._logger.handlers)):
            self._logger.addHandler(console_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
