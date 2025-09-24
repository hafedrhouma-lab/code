import torch
import torch.nn as nn
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.t5_transformer import T5Transformer
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.distelbert_transformer import DistilBERTTransformer
from projects.vendor_ranking.session_based_ranker.models.transformers_layers.transfomers_utils import create_transformer_config

class OfflineModel(nn.Module):
    def __init__(self, feature_configs, num_numerical_features=5, numerical_feat_hidd=32, numerical_feat_output=16, compressed_dim=2576):
        super(OfflineModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.transformers = nn.ModuleDict()
        self.compressed_dim = compressed_dim
        self.hidden_sizes = []

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

        self.numerical_feature_net = nn.Sequential(nn.Linear(num_numerical_features, numerical_feat_hidd),
                                            nn.BatchNorm1d(numerical_feat_hidd),
                                            nn.ReLU(),

                                            nn.Linear(numerical_feat_hidd, numerical_feat_output),
                                            nn.BatchNorm1d(numerical_feat_output),
                                            nn.ReLU())

        self.hidden_sizes.append(numerical_feat_output)
        self.combined_hidden_size = sum(self.hidden_sizes)

        # Check if compression is needed
        if self.combined_hidden_size != self.compressed_dim:
            self.compression_layer = nn.Sequential(
                nn.Linear(self.combined_hidden_size, self.compressed_dim * 2),
                nn.BatchNorm1d(self.compressed_dim * 2),
                nn.ReLU(),
                nn.Linear(self.compressed_dim * 2, self.compressed_dim),
                nn.BatchNorm1d(self.compressed_dim),
                nn.ReLU()
            )
            self.use_compression = True
        else:
            self.use_compression = False

            
    def forward(self, **inputs):
        hidden_states = []
        for feature_name, transformer in self.transformers.items():
            input_ids = inputs[f'{feature_name}_ids'].to(self.device)
            attention_mask = inputs[f'{feature_name}_mask'].to(self.device)
            last_hidden_state = transformer.get_last_hidden_state(input_ids, attention_mask)
            hidden_states.append(last_hidden_state)

        numerical_feat_emb = self.numerical_feature_net(inputs["numerical_features"].to(self.device))#.to(self.device))
        concatenated_features = torch.cat(hidden_states, dim=1)#.to(self.device)
        concatenated_features = torch.cat((concatenated_features, numerical_feat_emb), dim=1)#.to(self.device)

        if self.use_compression:
            concatenated_features = self.compression_layer(concatenated_features)

        return concatenated_features


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
                    
                
    def get_total_hidden_size(self):
        if self.use_compression:
            return self.compressed_dim
        return sum(self.hidden_sizes)
