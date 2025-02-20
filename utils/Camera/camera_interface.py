from abc import *


class CameraInterface(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_color(self):
        pass

    @abstractmethod
    def get_depth(self):
        pass

    @abstractmethod
    def get_color_depth(self):
        pass
