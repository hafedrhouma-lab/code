import time
from transformers import Trainer
from datasets import Dataset
import torch
import mlflow
import numpy as np

from projects.vendor_ranking.hf_transformers.utils.preprocess import (
    data_collator_refactored,
    preprocess_function_refactored,
    preprocess_test_input
)
from projects.vendor_ranking.hf_transformers.utils.training_utils import (
    get_tokenized_features,
    create_training_args
)
from projects.vendor_ranking.hf_transformers.utils.tokenizer import (
        create_tokenizer,
        create_fast_tokenizer,
        create_search_tokenizer,
        create_search_fast_tokenizer
    )
from projects.vendor_ranking.hf_transformers.utils.callbacks import (
        EvaluateEveryNStepsCallbackBucketized,
        EvaluateTrainEveryNStepsCallbackBucketized
    )
from projects.vendor_ranking.hf_transformers.utils.eval_utils import EvalModel
from projects.vendor_ranking.hf_transformers.models.t5_recommender import RecommenderClass
from projects.vendor_ranking.hf_transformers.utils.data_utils import (
        train_val_split_data,
        read_test_data_gcs_fs,
        clean_test_data
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(params, param_dates, data):
    log_dict = {}

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    log_dict["device_used"] = device
    
    print("device passed", params["device"])
    print("device used", device)

    if torch.cuda.is_available():
        print(f"CUDA is available. Torch is using CUDA version: {torch.version.cuda}")
        training_args_dict = params['training_args_gpu']
    else:
        print("CUDA is not available.")
        training_args_dict = params['training_args_cpu']

    print(f"PyTorch version: {torch.__version__}")

    nmb_train_samples = params["nmb_train_samples"]
    nmb_train_val_samples = params["nmb_train_val_samples"]
    nmb_val_samples = params["nmb_val_samples"]
    nmb_test_samples = params["nmb_test_samples"]

    freq_eval = params["freq_eval"]

    min_frequency = 3
    area_id_counts = data['delivery_area_id'].value_counts()
    geohash_counts = data['geohash6'].value_counts()

    valid_area_ids = area_id_counts[area_id_counts >= min_frequency].index
    valid_geohash = geohash_counts[geohash_counts >= min_frequency].index

    area_id_to_index = {area_id: index + 1 for index, area_id in enumerate(valid_area_ids)}
    geohash_to_index = {geohash: index + 1 for index, geohash in enumerate(valid_geohash)}

    area_id_to_index['unk'] = 0
    geohash_to_index['unk'] = 0

    train_data, _, train_val_data = train_val_split_data(
        data,
        nmb_train_samples,
        1,
        nmb_train_val_samples
    )
    val_data = read_test_data_gcs_fs(
        param_dates.test_date_start,
        param_dates.test_date_end,
        params["country_code"],
        nmb_val_samples,
        is_val=True
    )
    test_data = read_test_data_gcs_fs(
        param_dates.test_date_start,
        param_dates.test_date_end,
        params["country_code"],
        nmb_test_samples,
        is_val=False
    )

    print(f"Train Data Size: {train_data.shape[0]}")

    print(f"Train Validation Data Size: {train_val_data.shape[0]}")
    print(f"Validation Data Size: {val_data.shape[0]}")
    print(f"Test Data Size: {test_data.shape}")
    # Store data sizes in the logging dictionary
    log_dict["train_data_size"] = train_data.shape[0]
    log_dict["train_validation_data_size"] = train_val_data.shape[0]
    log_dict["validation_data_size"] = val_data.shape[0]
    log_dict["test_data_size"] = test_data.shape[0]

    # Prepare tokenizer
    chain_id_tokenizer_columns, max_chain_id_length = get_tokenized_features(params['tokenized_features'], tokenizer_type='chain_id')
    text_tokenizer_columns, max_text_length = get_tokenized_features(params['tokenized_features'], tokenizer_type='text')

    log_dict["tokenized_chain_id_columns"] = chain_id_tokenizer_columns
    log_dict["max_chain_id_length"] = max_chain_id_length
    log_dict["tokenized_text_columns"] = text_tokenizer_columns
    log_dict["max_text_length"] = max_text_length

    chain_id_tokenizer = create_tokenizer(
        data,
        vocab_size=data['chain_id'].nunique(),
        tokenizer_filename='chain_id_tokenizer.json',
        columns=chain_id_tokenizer_columns
    )
    fast_chain_id_tokenizer = create_fast_tokenizer(
        tokenizer_filename='chain_id_tokenizer.json',
        max_length=max_chain_id_length
    )

    t5_config_chain_id_dict = {
        "vocab_size": fast_chain_id_tokenizer.vocab_size,
        "d_model": params['t5_config_chain_id']['d_model'],
        "d_ff": params['t5_config_chain_id']['d_ff'],
        "num_layers": params['t5_config_chain_id']['num_layers'],
        "num_heads": params['t5_config_chain_id']['num_heads'],
        "dropout_rate": params['t5_config_chain_id']['dropout_rate'],
        "pad_token_id": fast_chain_id_tokenizer.pad_token_id,
        "decoder_start_token_id": fast_chain_id_tokenizer.pad_token_id
    }

    if len(text_tokenizer_columns) != 0:
        text_tokenizer = create_search_tokenizer(
            data,
            tokenizer_filename='text_tokenizer.json',
            column=text_tokenizer_columns[0]
        )
        fast_text_tokenizer = create_search_fast_tokenizer(
            tokenizer_filename='text_tokenizer.json',
            max_length=max_text_length
        )
        t5_config_text_dict = {
            "vocab_size": fast_text_tokenizer.vocab_size,
            "d_model": params['t5_config_text']['d_model'],
            "d_ff": params['t5_config_text']['d_ff'],
            "num_layers": params['t5_config_text']['num_layers'],
            "num_heads": params['t5_config_text']['num_heads'],
            "dropout_rate": params['t5_config_text']['dropout_rate'],
            "pad_token_id": fast_text_tokenizer.pad_token_id,
            "decoder_start_token_id": fast_text_tokenizer.pad_token_id
        }
    else:
        fast_text_tokenizer = None
        t5_config_text_dict = None

    fast_tokenizers = {
        "chain_id": fast_chain_id_tokenizer,
        "text": fast_text_tokenizer
    }

    t5_config_dicts = {
        "chain_id": t5_config_chain_id_dict,
        "text": t5_config_text_dict
    }

    # Build dynamic feature configuration based on config file
    feature_configs = []
    for feature in params['tokenized_features']:
        feature_config = {
            "name": feature['name'],
            "tokenizer": fast_tokenizers[feature['tokenizer']],
            "max_length": feature['max_length'],
            "model_type": feature['model_type'],
            "t5_config": t5_config_dicts[feature['tokenizer']]
        }
        feature_configs.append(feature_config)

    numerical_feature_names = params["numerical_features"]

    chain_id_vocab = fast_chain_id_tokenizer.get_vocab()

    model_artifacts = {
        'area_id_to_index': area_id_to_index,
        'geohash_to_index': geohash_to_index,
        'chain_id_vocab': chain_id_vocab
    }

    print(f"Number of unique 'chain_id' in original val_data: {val_data['chain_id'].nunique()}")
    log_dict["unique_chain_original_val_data"] = val_data['chain_id'].nunique()
    val_data = clean_test_data(val_data, model_artifacts)

    print(f"Number of unique 'chain_id' in train_data: {train_data['chain_id'].nunique()}")
    print(f"Number of unique 'chain_id' in train_val_data: {train_val_data['chain_id'].nunique()}")
    print(f"Number of unique 'chain_id' in cleaned val_data: {val_data['chain_id'].nunique()}")
    print(f"Validation Data Size after clean up: {val_data.shape[0]}")
    log_dict["unique_chain_train_data"] = train_data['chain_id'].nunique()
    log_dict["unique_chain_train_val_data"] = train_val_data['chain_id'].nunique()
    log_dict["unique_chain_cleaned_val_data"] = val_data['chain_id'].nunique()
    log_dict["validation_data_size_cleaned"] = val_data.shape[0]

    print(f"Number of unique 'chain_id' in original test_data: {test_data['chain_id'].nunique()}")
    log_dict["unique_chain_original_test_data"] = test_data['chain_id'].nunique()
    test_data = clean_test_data(test_data, model_artifacts)

    print(f"Number of unique 'chain_id' in cleaned test_data: {test_data['chain_id'].nunique()}")
    print(f"Test Data Size after clean up: {test_data.shape[0]}")
    log_dict["unique_chain_cleaned_test_data"] = test_data['chain_id'].nunique()
    log_dict["test_data_size_cleaned"] = test_data.shape[0]

    train_dataset = Dataset.from_pandas(train_data)
    train_val_dataset = Dataset.from_pandas(train_val_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)


    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                         feature_configs=feature_configs,
                                                                                         area_id_to_index=area_id_to_index,
                                                                                         geohash_to_index=geohash_to_index,
                                                                                         numerical_feat_names=numerical_feature_names,
                                                                                         label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True, num_proc=1)
    tokenized_train_val_dataset = train_val_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                         feature_configs=feature_configs,
                                                                                         area_id_to_index=area_id_to_index,
                                                                                         geohash_to_index=geohash_to_index,
                                                                                         numerical_feat_names=numerical_feature_names,
                                                                                         label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True)
    tokenized_val_dataset = val_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                         feature_configs=feature_configs,
                                                                                         area_id_to_index=area_id_to_index,
                                                                                         geohash_to_index=geohash_to_index,
                                                                                         numerical_feat_names=numerical_feature_names,
                                                                                         label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True)
    tokenized_test_dataset = test_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                     feature_configs=feature_configs,
                                                                                     area_id_to_index=area_id_to_index,
                                                                                     geohash_to_index=geohash_to_index,
                                                                                     numerical_feat_names=numerical_feature_names,
                                                                                     label_tokenizer=fast_chain_id_tokenizer),
                                            batched=True)


    input_vocab_sizes = [config['tokenizer'].vocab_size for config in feature_configs]
    output_vocab_size = fast_chain_id_tokenizer.vocab_size
    log_dict["input_vocab_sizes"] = input_vocab_sizes
    log_dict["output_vocab_size"] = output_vocab_size


    # Initialize the model
    model = RecommenderClass(
        feature_configs=feature_configs,
        numerical_features=numerical_feature_names,
        order_hour_dim=params['order_hour_dim'],
        area_id_vocab_size=len(area_id_to_index),
        area_id_dim=params['area_id_dim'],
        geohash_vocab_size=len(geohash_to_index),
        geohash_dim=params['geohash_dim'],
        output_vocab_size=output_vocab_size,
        chain_id_vocab=chain_id_vocab,
        area_id_to_index=area_id_to_index,
        geohash_to_index=geohash_to_index,
        compressed_dim=params['compressed_dim'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    # Set training arguments
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     save_total_limit=5,
    #     save_steps=1000,
    #     evaluation_strategy="epoch",
    #     save_strategy='no',
    #     learning_rate=2e-4,
    #     per_device_train_batch_size=1024,
    #     per_device_eval_batch_size=200,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     remove_unused_columns=False
    # )

    training_args = create_training_args(training_args_dict)

    # Initialize callback
    eval_callback = EvaluateEveryNStepsCallbackBucketized(eval_steps=freq_eval,
                                                          feature_configs=feature_configs,
                                                          label_tokenizer=fast_chain_id_tokenizer,
                                                          area_id_to_index=area_id_to_index,
                                                          geohash_to_index=geohash_to_index,
                                                          numerical_feat_names=numerical_feature_names
                                                          )
    train_eval_callback = EvaluateTrainEveryNStepsCallbackBucketized(eval_steps=freq_eval,
                                                          feature_configs=feature_configs,
                                                          label_tokenizer=fast_chain_id_tokenizer,
                                                          area_id_to_index=area_id_to_index,
                                                          geohash_to_index=geohash_to_index,
                                                          numerical_feat_names=numerical_feature_names,
                                                          eval_data=tokenized_train_val_dataset
                                                          )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=lambda features: data_collator_refactored(features, feature_configs),
        callbacks=[eval_callback, train_eval_callback]
    )

    # Set the trainer instance in the callback
    eval_callback.trainer = trainer
    train_eval_callback.trainer = trainer

    # logging parameters to MLflow before training
    for key, value in log_dict.items():
        mlflow.log_param(key, value)

    # Train the model
    trainer.train()
    print("model completed training")

    eval_metrics = eval_callback.metrics
    print("Evaluation metrics during training:")
    print(eval_metrics)

    eval_model_instance = EvalModel(test_batch_size=params['test_batch_size'],
                                                                          feature_configs=feature_configs,
                                                                          label_tokenizer=fast_chain_id_tokenizer,
                                                                          area_id_to_index=area_id_to_index,
                                                                          geohash_to_index=geohash_to_index,
                                                                          numerical_feat_names=numerical_feature_names
                                                                          )
    test_metrics = eval_model_instance.eval_model_data(model, tokenized_test_dataset)

    print("Test metrics after training:")
    print(test_metrics)

    n = model.offline_model.get_total_hidden_size()
    float32_array = np.random.rand(n).astype(np.float32)
    offline_embedding = float32_array.tolist()
    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20]
    }

    model.eval()

    # device = params['device']
    model.to(device)
    print("model device:", device)

    model_input = preprocess_test_input(
        model_input,
        device
    )
    start_time = time.time()
    with torch.no_grad():
        _, predicted_logits = model.online_model(
            model_input['offline_embedding'],
            model_input['order_hour'],
            model_input['delivery_area_id'],
            model_input['geohash6']
        )
    time_taken = time.time() - start_time
    print(f"Time taken on device {device} : {time_taken}")
    mlflow.log_metric(f"inference_time_{device}", time_taken)

    device = "cpu"
    model.to(device)
    print("model device:", device)

    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20]
    }

    model_input = preprocess_test_input(
        model_input,
        device
    )
    start_time = time.time()
    with torch.no_grad():
        _, predicted_logits = model.online_model(
            model_input['offline_embedding'],
            model_input['order_hour'],
            model_input['delivery_area_id'],
            model_input['geohash6']
        )
    time_taken = time.time() - start_time
    print(f"Time taken on device {device} : {time_taken}")
    mlflow.log_metric(f"inference_time_{device}", time_taken)

    recall_10 = round(test_metrics['Test_Recall_10'], 4)
    print(f"Recall at 10 {recall_10}")

    return model, fast_chain_id_tokenizer, recall_10