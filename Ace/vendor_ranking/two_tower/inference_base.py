from abc import ABC, abstractmethod


class BaseInferenceModel(ABC):
    @abstractmethod
    def get_chains_embeddings(self):
        pass

    @abstractmethod
    def get_user_embeddings(self, user_features):
        pass

    @abstractmethod
    def get_sorted_chains(self, user_features, available_chains):
        pass
