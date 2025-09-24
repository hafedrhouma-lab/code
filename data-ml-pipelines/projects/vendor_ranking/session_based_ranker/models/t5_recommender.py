import os
import torch
import torch.nn as nn
from projects.vendor_ranking.session_based_ranker.models.offline_model import OfflineModel
from projects.vendor_ranking.session_based_ranker.models.online_model import OnlineModel
import json

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecommenderClass(nn.Module):
    def __init__(self, feature_configs, online_feature_configs, numerical_features, order_hour_dim, area_id_vocab_size, area_id_dim,
                 geohash_vocab_size, geohash_dim, output_vocab_size, chain_id_vocab, num_numerical_features=5,
                 area_id_to_index=None, geohash_to_index=None, compressed_dim=2576, dropout_rate=0.2,
                 quantize_model=False, compile_model=False):
        super(RecommenderClass, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.quantize_model = quantize_model
        self.compile_model = compile_model
        self.feature_configs = feature_configs
        self.online_feature_configs = online_feature_configs
        self.numerical_features = numerical_features
        self.offline_model = OfflineModel(feature_configs, num_numerical_features=len(numerical_features),
                                          compressed_dim=compressed_dim).to(self.device)

        self.area_id_to_index = area_id_to_index
        self.geohash_to_index = geohash_to_index
        self.chain_id_vocab = chain_id_vocab

        total_hidden_size = self.offline_model.get_total_hidden_size()

        self.online_model = OnlineModel(
            offline_output_dim=total_hidden_size,
            order_hour_vocab_size=24,
            order_hour_dim=order_hour_dim,
            area_id_vocab_size=area_id_vocab_size,
            area_id_dim=area_id_dim,
            geohash_vocab_size=geohash_vocab_size,
            geohash_dim=geohash_dim,
            vocab_size=output_vocab_size,
            dropout_rate=dropout_rate,
            feature_configs=online_feature_configs,
        ).to(self.device)

    def forward(self, **inputs):
        offline_inputs = {key: value.to(self.device) for key, value in inputs.items() if
                          key.endswith('_ids') or key.endswith('_mask') or key=="numerical_features"}

        order_hour = inputs.get('order_hour').to(self.device)
        delivery_area_id = inputs.get('delivery_area_id').to(self.device)
        geohash = inputs.get('geohash6').to(self.device)

        # Get concatenated output from the offline model
        offline_output = self.offline_model(**offline_inputs)

        if 'labels' in inputs:
            labels = inputs.get('labels').to(self.device)
            loss, logits = self.online_model(offline_output, order_hour, delivery_area_id, geohash, labels=labels, **offline_inputs)
        else:
            loss, logits = self.online_model(offline_output, order_hour, delivery_area_id, geohash, labels=None, **offline_inputs)

        return loss, logits

            
    def load_model_to_device(self, device):
        self.device = device
        self.offline_model.device = device
        self.online_model.device = device

        self.offline_model = self.offline_model.to(device)
        self.online_model = self.online_model.to(device)

        
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
        offline_model_on_device = all(param.device == device for param in self.offline_model.parameters())
        # Check if all parameters of the online_model are on the correct device
        online_model_on_device = all(param.device == device for param in self.online_model.parameters())

        # Check if self.device matches the target device
        self_device_correct = self.device == device
        offline_device_correct = self.offline_model.device == device
        online_device_correct = self.online_model == device

        # Return True only if all checks pass
        return offline_model_on_device and online_model_on_device and self_device_correct and offline_device_correct and online_device_correct
            
    
    def save_models(self, path):
        offline_path = os.path.join(path, "offline_model")
        online_path = os.path.join(path, "online_model")
        config_path = os.path.join(path, "model_config.json")
        tokenizer_path = os.path.join(path, "tokenizers")
        area_id_mapping_path = os.path.join(path, "area_id_to_index.json")
        geohash_mapping_path = os.path.join(path, "geohash_to_index.json")
        chain_id_vocab_path = os.path.join(path, "chain_id_vocab.json")

        os.makedirs(offline_path, exist_ok=True)
        os.makedirs(online_path, exist_ok=True)
        os.makedirs(tokenizer_path, exist_ok=True)

        self.offline_model.to(torch.device("cpu"))
        self.online_model.to(torch.device("cpu"))

        '''
        ### Maybe add quantization here based on model_config features ? 
        if self.quantize_model:
            self.online_model = quantize_dynamic(
                self.online_model,  # The model to quantize
                {torch.nn.Linear},  # Layers to quantize (e.g., Linear layers)
                dtype=torch.qint8   # Quantized dtype
            )

            self.offline_model = quantize_dynamic(
                self.offline_model,  # The model to quantize
                {torch.nn.Linear},  # Layers to quantize (e.g., Linear layers)
                dtype=torch.qint8   # Quantized dtype
            )
        '''
        
        torch.save(self.offline_model.state_dict(), os.path.join(offline_path, "offline_pytorch_model.bin"))
        torch.save(self.online_model.state_dict(), os.path.join(online_path, "online_pytorch_model.bin"))

        # Save tokenizers and update feature configs
        for feature_config in self.feature_configs:
            tokenizer_file = os.path.join(tokenizer_path, f"{feature_config['name']}_tokenizer")
            feature_config['tokenizer'].save_pretrained(tokenizer_file)
            feature_config['tokenizer_file'] = tokenizer_file
            del feature_config['tokenizer']  # Remove tokenizer from the config as it can't be serialized

        for feature_config in self.online_feature_configs:
            tokenizer_file = os.path.join(tokenizer_path, f"{feature_config['name']}_tokenizer")
            feature_config['tokenizer'].save_pretrained(tokenizer_file)
            feature_config['tokenizer_file'] = tokenizer_file
            del feature_config['tokenizer']  # Remove tokenizer from the config as it can't be serialized

        # Save the config along with the models
        model_config = {
            "feature_configs": self.feature_configs,
            "online_feature_configs": self.online_feature_configs,
            "numerical_features": self.numerical_features,
            "order_hour_dim": self.online_model.order_hour_embedding.embedding_dim,
            "area_id_vocab_size": self.online_model.area_id_embedding.num_embeddings,
            "area_id_dim": self.online_model.area_id_embedding.embedding_dim,
            "geohash_vocab_size": self.online_model.geohash_embedding.num_embeddings,
            "geohash_dim": self.online_model.geohash_embedding.embedding_dim,
            "output_vocab_size": self.online_model.final_layer.out_features,
            "offline_hidden_size": self.offline_model.get_total_hidden_size(),
            "quantize_model": self.quantize_model,
            "compile_model": self.compile_model,
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f)

        # area_id_to_index_serializable = {int(key): value for key, value in self.area_id_to_index.items()}
        area_id_to_index_serializable = {str(key): value for key, value in self.area_id_to_index.items()}
        with open(area_id_mapping_path, 'w') as f:
            json.dump(area_id_to_index_serializable, f)

        geohash_to_index_serializable = {key: value for key, value in self.geohash_to_index.items()}
        with open(geohash_mapping_path, 'w') as f:
            json.dump(geohash_to_index_serializable, f)

        with open(chain_id_vocab_path, 'w') as f:
            json.dump(self.chain_id_vocab, f)

    ### Are the following 3 functions used anywhere actually ? double check with Anas, as I found a few issues here in the original versions
    @staticmethod
    def load_offline_model(path):
        config_path = os.path.join(path, "model_config.json")
        offline_path = os.path.join(path, "offline_model", "offline_pytorch_model.bin")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        if model_config['use_sess_clicks_online']:
            ## Create New Temp Feature Configs WITHOUT session_clicks
            temp_feature_configs = [config for config in model_config['feature_configs'] if config['name'] != "session_clicks"]
            offline_model = OfflineModel(
                feature_configs=temp_feature_configs,
                num_numerical_features=len(model_config["numerical_features"]),
                compressed_dim=model_config["offline_hidden_size"]
            )
        else:

            offline_model = OfflineModel(
                feature_configs=model_config["feature_configs"],
                num_numerical_features=len(model_config["numerical_features"]),
                compressed_dim=model_config["offline_hidden_size"]
            )
            
        offline_model.load_state_dict(torch.load(offline_path, weights_only=True))

        return offline_model

    @staticmethod
    def load_online_model(path):
        config_path = os.path.join(path, "model_config.json")
        online_path = os.path.join(path, "online_model", "online_pytorch_model.bin")
        area_id_mapping_path = os.path.join(path, "area_id_to_index.json")
        chain_id_vocab_path = os.path.join(path, "chain_id_vocab.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        with open(area_id_mapping_path, 'r') as f:
            area_id_to_index = json.load(f)

        # Convert the keys back to integers
        area_id_to_index = {int(key): value for key, value in area_id_to_index.items()}

        with open(chain_id_vocab_path, 'r') as f:
            chain_id_vocab = json.load(f)
            
        if model_config['use_sess_clicks_online']:
            temp_feature_configs = [config for config in model_config['feature_configs'] if config['name'] == "session_clicks"]

            online_model = OnlineModel(
                offline_output_dim=model_config["offline_hidden_size"],
                order_hour_vocab_size=24,
                order_hour_dim=model_config["order_hour_dim"],
                area_id_vocab_size=model_config["area_id_vocab_size"],
                area_id_dim=model_config["area_id_dim"],
                geohash_vocab_size=model_config["geohash_vocab_size"],
                geohash_dim=model_config["geohash_dim"],
                vocab_size=model_config["output_vocab_size"],
                feature_configs = temp_feature_configs,
            )


        else:
            online_model = OnlineModel(
                offline_output_dim=model_config["offline_hidden_size"],
                order_hour_vocab_size=24,
                order_hour_dim=model_config["order_hour_dim"],
                area_id_vocab_size=model_config["area_id_vocab_size"],
                area_id_dim=model_config["area_id_dim"],
                geohash_vocab_size=model_config["geohash_vocab_size"],
                geohash_dim=model_config["geohash_dim"],
                vocab_size=model_config["output_vocab_size"]
            )

            
            
        online_model.load_state_dict(torch.load(online_path, weights_only=True))

        return online_model, area_id_to_index, chain_id_vocab

    @staticmethod
    def load_model(path):
        config_path = os.path.join(path, "model_config.json")
        area_id_mapping_path = os.path.join(path, "area_id_to_index.json")
        chain_id_vocab_path = os.path.join(path, "chain_id_vocab.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        with open(area_id_mapping_path, 'r') as f:
            area_id_to_index = json.load(f)
        # Convert the keys back to integers
        area_id_to_index = {int(key): value for key, value in area_id_to_index.items()}

        with open(chain_id_vocab_path, 'r') as f:
            chain_id_vocab = json.load(f)

        recommender = RecommenderClass(
            feature_configs=model_config["feature_configs"],
            numerical_features=model_config["numerical_features"],
            order_hour_dim=model_config["order_hour_dim"],
            area_id_vocab_size=model_config["area_id_vocab_size"],
            area_id_dim=model_config["area_id_dim"],
            geohash_vocab_size=model_config["geohash_vocab_size"],
            geohash_dim=model_config["geohash_dim"],
            output_vocab_size=model_config["output_vocab_size"],
            chain_id_vocab=chain_id_vocab,
            area_id_to_index=area_id_to_index,
            geohash_to_index=geohash_to_index,
            compressed_dim=model_config["offline_hidden_size"],
            sess_clicks_online = model_config["use_sess_clicks_online"]
        )
        
        
        recommender.offline_model.load_state_dict(torch.load(os.path.join(path, "offline_model", "offline_pytorch_model.bin"),
                                                             weights_only=True))
        recommender.online_model.load_state_dict(torch.load(os.path.join(path, "online_model", "online_pytorch_model.bin"),
                                                                         weights_only=True))

        return recommender

    
    
