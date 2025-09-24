from transformers import T5ForConditionalGeneration, T5Model
from projects.vendor_ranking.hf_transformers.models.transformers_layers.base_transformer import TransformerBase
import torch
class T5Transformer(TransformerBase):
    def __init__(self, config):
        super(T5Transformer, self).__init__(config)
        self.model = T5ForConditionalGeneration(config)

    def get_last_hidden_state(self, input_ids, attention_mask):
        decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).expand(input_ids.size(0), -1).to(input_ids.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return outputs.encoder_last_hidden_state[:, -1, :]

    def get_hidden_size(self):
        return self.model.config.d_model
