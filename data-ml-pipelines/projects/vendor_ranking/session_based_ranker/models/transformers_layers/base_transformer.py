import torch
import torch.nn as nn

class TransformerBase(nn.Module):
    def __init__(self, config):
        super(TransformerBase, self).__init__()
        self.config = config

    def get_last_hidden_state(self, input_ids, attention_mask):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_hidden_size(self):
        raise NotImplementedError("Subclasses should implement this method.")