import torch
import torch.nn as nn
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.t5_transformer import T5Transformer
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.distelbert_transformer import DistilBERTTransformer
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.transfomers_utils import create_transformer_config


class OnlineModel(nn.Module):
    def __init__(self, offline_output_dim, order_hour_vocab_size, order_hour_dim,
                 area_id_vocab_size, area_id_dim, geohash_vocab_size, geohash_dim, vocab_size, dropout_rate=0.2,
                 feature_configs=None):
        super(OnlineModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.transformers = nn.ModuleDict()
        self.hidden_sizes = []
        self.feature_configs = feature_configs

        # Embeddings for online features
        self.order_hour_embedding = nn.Embedding(order_hour_vocab_size, order_hour_dim)
        self.area_id_embedding = nn.Embedding(area_id_vocab_size, area_id_dim)
        self.geohash_embedding = nn.Embedding(geohash_vocab_size, geohash_dim)

        for idx, feature_config in enumerate(feature_configs):
            name = feature_config["name"]
            model_type = feature_config["model_type"]
            t5_config = feature_config['t5_config']

            config = create_transformer_config(model_type, t5_config)

            if model_type == 'T5':
                transformer = T5Transformer(config).to(self.device)
            elif model_type == 'DistilBERT':
                transformer = DistilBERTTransformer(config).to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.transformers[name] = transformer
            self.hidden_sizes.append(transformer.get_hidden_size())

        self.hidden_sizes.extend([order_hour_dim, area_id_dim, geohash_dim])
        self.combined_hidden_size = offline_output_dim + sum(self.hidden_sizes)

        self.interaction_layer = nn.Sequential(
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU()
        )

        # Final linear layer to produce the logits
        self.final_layer = nn.Linear(self.combined_hidden_size, vocab_size)

        self.dropout = nn.Dropout(p=dropout_rate)
 
    def forward(self, offline_output, order_hour, delivery_area_id, geohash, labels=None, **inputs):
        device = offline_output.device
        hidden_states = []

        for feature_name, transformer in self.transformers.items():
            input_ids = inputs[f'{feature_name}_ids'].to(device)
            attention_mask = inputs[f'{feature_name}_mask'].to(device)
            last_hidden_state = transformer.get_last_hidden_state(input_ids, attention_mask)
            hidden_states.append(last_hidden_state)

        order_hour_emb = self.order_hour_embedding(order_hour)
        area_id_emb = self.area_id_embedding(delivery_area_id)
        geohash_emb = self.geohash_embedding(geohash)

        # Ensure the dimensions are compatible for concatenation
        if len(order_hour_emb.shape) == 3:
            order_hour_emb = order_hour_emb.squeeze(1)
        if len(area_id_emb.shape) == 3:
            area_id_emb = area_id_emb.squeeze(1)
        if len(geohash_emb.shape) == 3:
            geohash_emb = geohash_emb.squeeze(1)

        concatenated_features = torch.cat(hidden_states, dim=1)#.to(device)
        combined_features = torch.cat((offline_output, order_hour_emb, area_id_emb, geohash_emb, concatenated_features), dim=1)

        combined_features = self.dropout(combined_features)

        combined_features = self.interaction_layer(combined_features)

        logits = self.final_layer(combined_features).to(device)

        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(device))

        return loss, logits

    def set_model_device(self, device):
        self.device = device

    def confirm_model_on_device(self, device):
        """
        Confirm that all parameters of offline_model and online_model,
        as well as self.device, are on the specified device.

        Args:
            device (torch.device): The device to check against.

        Returns:
            bool: True if all parameters of the models and self.device are on the specified device, False otherwise.
        """
        # Check if all parameters of the offline_model are on the correct device
        offline_model_on_device = all(param.device == device for param in self.parameters())
        self_device_correct = self.device == device
        return offline_model_on_device and self_device_correct
               