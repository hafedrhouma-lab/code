"""Definition of custom logger class"""
import logging
import sys


class Logger:
    """class logger"""

    def __init__(self, logger_name: str = __name__):
        """Custom logger class
        Args:
            logger_name (str, optional): name of logger. Defaults to __name__.
        """
        self._formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        self._level = logging.DEBUG
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.setLevel(self._level)

    @property
    def formatter(self):
        """property - get format"""
        return self._formatter

    @formatter.setter
    def formatter(self, new_format: str):
        """property - set formatter
        Args:
            new_format (str): new format
        """
        self._formatter = logging.Formatter(new_format)

    @property
    def log_level(self):
        """property - get log_level"""
        return self._level

    @log_level.setter
    def log_level(self, new_log_level):
        """property - set log_level
        Args:
            new_log_level (logging): new logging level
        """
        self._level = new_log_level
        self.logger.setLevel(self._level)

    def _get_console_handler(self):
        """Get logging stdout handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._formatter)
        console_handler.setLevel(logging.INFO)
        return console_handler

    def _get_file_handler(self, log_file: str):
        """Get logging file handler
        Args:
            log_file (str): filename to store logging info
        """
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(self._formatter)
        file_handler.setLevel(logging.INFO)
        return file_handler

    def get_logger(
        self,
        add_file_handler: str = "",
        add_stdout_handler: bool = True,
        remove_exist_handler: bool = False,
    ):
        """Define logger with different handler
        Args:
            add_file_handler (str, optional): filename to store logging info. Defaults to "".
            add_stdout_handler (bool, optional): stdout handler identifier. Defaults to True.
            remove_exist_handler (bool, optional): whether to remove all the existing handlers. Defaults to False.
        Returns:
            logger with handler
        """

        if remove_exist_handler:
            self.logger.handlers.clear()

        exist_handlers = [
            handler.__class__.__name__ for handler in self.logger.handlers
        ]

        if (
            not self.logger.hasHandlers() or "StreamHandler" not in exist_handlers
        ) and add_stdout_handler:
            self.logger.addHandler(self._get_console_handler())

        if (
            not self.logger.hasHandlers() or "FileHandler" not in exist_handlers
        ) and add_file_handler:
            self.logger.addHandler(self._get_file_handler(add_file_handler))

        return self.logger
