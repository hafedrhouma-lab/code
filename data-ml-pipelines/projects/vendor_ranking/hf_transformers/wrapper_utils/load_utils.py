import torch
import json
import os

from projects.vendor_ranking.hf_transformers.models.offline_model import OfflineModel
from projects.vendor_ranking.hf_transformers.models.online_model import OnlineModel
from projects.vendor_ranking.hf_transformers.models.t5_recommender import RecommenderClass
from transformers import PreTrainedTokenizerFast



def load_offline_model(model_config, model_path):
    offline_model = OfflineModel(
        feature_configs=model_config["feature_configs"],
        num_numerical_features=len(model_config["numerical_features"]),
        compressed_dim=model_config["offline_hidden_size"]
    )
    offline_model.load_state_dict(torch.load(model_path, weights_only=True))

    return offline_model


def load_online_model(model_config, model_path, area_id_mapping_path, geohash_to_index_path, chain_id_vocab_path):
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
    online_model.load_state_dict(torch.load(model_path, weights_only=True))

    with open(area_id_mapping_path, 'r') as f:
        area_id_to_index = json.load(f)

    with open(geohash_to_index_path, 'r') as f:
        geohash_to_index = json.load(f)

    with open(chain_id_vocab_path, 'r') as f:
        chain_id_vocab = json.load(f)

    return online_model, area_id_to_index, geohash_to_index, chain_id_vocab


def load_model(model_config, offline_model_path, online_model_path, area_id_mapping_path, geohash_to_index_path, chain_id_vocab_path):
    with open(area_id_mapping_path, 'r') as f:
        area_id_to_index = json.load(f)

    with open(geohash_to_index_path, 'r') as f:
        geohash_to_index = json.load(f)

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
        compressed_dim=model_config["offline_hidden_size"]
    )

    recommender.offline_model.load_state_dict(torch.load(offline_model_path, weights_only=True))
    recommender.online_model.load_state_dict(torch.load(online_model_path, weights_only=True))

    return recommender, area_id_to_index, geohash_to_index, chain_id_vocab


def load_feature_tokenizers(feature_configs, tokenizer_path):
    for feature_config in feature_configs:
        tokenizer_file = os.path.join(tokenizer_path, f"{feature_config['name']}_tokenizer")
        feature_config['tokenizer'] = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
