from abc import ABC
from datetime import datetime
from pathlib import Path


class Logger(ABC):
    '''
    Class representing a base logger
    '''
    def log(self, message: str) -> None:
        pass


class ConsoleLogger(Logger):
    def log(self, message: str) -> None:
        print(f'[{datetime.now():%H:%M:%S}] {message}')


class FileLogger(Logger):
    def __init__(self, file: Path) -> None:
        self.filepath = file

    def log(self, message: str) -> None:
        with open(str(self.filepath), 'wa') as file:
            file.write(f'[{datetime.now():%H:%M:%S}] {message}\n')


class MultipleLogger(Logger):
    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers

    def log(self, message: str) -> None:
        for logger in self.loggers:
            logger.log(message)
