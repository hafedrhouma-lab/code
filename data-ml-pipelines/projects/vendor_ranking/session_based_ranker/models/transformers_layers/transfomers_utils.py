from transformers import T5Config, DistilBertConfig


def create_transformer_config(model_type, t5_config_dict):
    if model_type == "T5":
        return T5Config(
            vocab_size=t5_config_dict["vocab_size"],
            d_model=t5_config_dict["d_model"],
            d_ff=t5_config_dict["d_ff"],
            num_layers=t5_config_dict["num_layers"],
            num_heads=t5_config_dict["num_heads"],
            dropout_rate=t5_config_dict["dropout_rate"],
            # these 2 are changed from 0
            pad_token_id=t5_config_dict["pad_token_id"],
            decoder_start_token_id=t5_config_dict["decoder_start_token_id"]
        )
    elif model_type == "DistilBERT":
        return DistilBertConfig(
            vocab_size=t5_config_dict["vocab_size"],
            dim=512,
            hidden_dim=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            pad_token_id=0,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
