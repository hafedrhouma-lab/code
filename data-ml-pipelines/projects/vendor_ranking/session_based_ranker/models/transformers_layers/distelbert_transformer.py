from projects.vendor_ranking.session_based_ranker.models.transformers_layers.base_transformer import TransformerBase
from transformers import DistilBertModel

class DistilBERTTransformer(TransformerBase):
    def __init__(self, config):
        super(DistilBERTTransformer, self).__init__(config)
        self.model = DistilBertModel(config)

    def get_last_hidden_state(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, -1, :]

    def get_hidden_size(self):
        return self.model.config.dim

