"""Declaration of IModel interfaces"""
from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def fit(self):
        """Fit classifier on dataset"""
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """Predict on dataset"""
        raise NotImplementedError

    @abstractmethod
    def save(self, version):
        """Save classifier object to given file path

        Args:
            version ([type]): object version
            path ([type]): file path

        Raises:
            NotImplementedError: raise if not implemented
        """

        raise NotImplementedError

    @abstractmethod
    def load(self, version):
        """Load classifier object from a given file path

        Args:
            version ([type]): object version
            path ([type]): file path

        Raises:
            NotImplementedError: raise if not implemented
        """
        raise NotImplementedError
