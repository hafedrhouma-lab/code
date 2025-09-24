import os
import torch
import torch.nn as nn
from projects.vendor_ranking.hf_transformers.models.offline_model import OfflineModel
from projects.vendor_ranking.hf_transformers.models.online_model import OnlineModel
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecommenderClass(nn.Module):
    def __init__(self, feature_configs, numerical_features, order_hour_dim, area_id_vocab_size, area_id_dim,
                 geohash_vocab_size, geohash_dim, output_vocab_size, chain_id_vocab,
                 num_numerical_features=5, area_id_to_index=None, geohash_to_index=None, compressed_dim=2576, dropout_rate=0.2):
        super(RecommenderClass, self).__init__()
        self.device = device
        self.feature_configs = feature_configs
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
            vocab_size=output_vocab_size,  # Use the output vocab size here
            dropout_rate=dropout_rate
        ).to(self.device)

    def forward(self, **inputs):
        offline_inputs = {key: value.to(self.device) for key, value in inputs.items() if
                          key.endswith('_ids') or key.endswith('_mask') or key=="numerical_features"} #TODO: make sure it passes it

        order_hour = inputs.get('order_hour').to(self.device)
        delivery_area_id = inputs.get('delivery_area_id').to(self.device)
        geohash = inputs.get('geohash6').to(self.device)

        # Get concatenated output from the offline model
        offline_output = self.offline_model(**offline_inputs)

        if 'labels' in inputs:
            labels = inputs.get('labels').to(self.device)
            # loss, logits = self.online_model(offline_output, order_hour, order_weekday, delivery_area_id, geohash, labels)
            loss, logits = self.online_model(offline_output, order_hour, delivery_area_id, geohash, labels)
        else:
            # loss, logits = self.online_model(offline_output, order_hour, order_weekday, delivery_area_id, geohash)
            loss, logits = self.online_model(offline_output, order_hour, delivery_area_id, geohash)

        return loss, logits

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

        torch.save(self.offline_model.state_dict(), os.path.join(offline_path, "offline_pytorch_model.bin"))
        torch.save(self.online_model.state_dict(), os.path.join(online_path, "online_pytorch_model.bin"))

        # Save tokenizers and update feature configs
        for feature_config in self.feature_configs:
            tokenizer_file = os.path.join(tokenizer_path, f"{feature_config['name']}_tokenizer")
            feature_config['tokenizer'].save_pretrained(tokenizer_file)
            feature_config['tokenizer_file'] = tokenizer_file
            del feature_config['tokenizer']  # Remove tokenizer from the config as it can't be serialized

        # Save the config along with the models
        model_config = {
            "feature_configs": self.feature_configs,
            "numerical_features": self.numerical_features,
            "order_hour_dim": self.online_model.order_hour_embedding.embedding_dim,
            "area_id_vocab_size": self.online_model.area_id_embedding.num_embeddings,
            "area_id_dim": self.online_model.area_id_embedding.embedding_dim,
            "geohash_vocab_size": self.online_model.geohash_embedding.num_embeddings,
            "geohash_dim": self.online_model.geohash_embedding.embedding_dim,
            "output_vocab_size": self.online_model.final_layer.out_features,
            "offline_hidden_size": self.offline_model.get_total_hidden_size()
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

    @staticmethod
    def load_offline_model(path):
        config_path = os.path.join(path, "model_config.json")
        offline_path = os.path.join(path, "offline_model", "offline_pytorch_model.bin")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        offline_model = OfflineModel(model_config["feature_configs"], model_config["input_vocab_sizes"])
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

        online_model = OnlineModel(
            offline_output_dim=model_config["offline_hidden_size"],
            order_hour_vocab_size=24,
            order_hour_dim=model_config["order_hour_dim"],
            area_id_vocab_size=model_config["area_id_vocab_size"],
            area_id_dim=model_config["area_id_dim"],
            vocab_size=model_config["output_vocab_size"]  # Load the output vocab size here
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
            order_hour_dim=model_config["order_hour_dim"],
            area_id_vocab_size=model_config["area_id_vocab_size"],
            area_id_dim=model_config["area_id_dim"],
            input_vocab_sizes=model_config["input_vocab_sizes"],
            output_vocab_size=model_config["output_vocab_size"],
            chain_id_vocab=chain_id_vocab,
            area_id_to_index=area_id_to_index
        )

        recommender.offline_model.load_state_dict(torch.load(os.path.join(path, "offline_model", "offline_pytorch_model.bin"),
                                                             weights_only=True))
        recommender.online_model.load_state_dict(torch.load(os.path.join(path, "online_model", "online_pytorch_model.bin"),
                                                                         weights_only=True))

        return recommender
