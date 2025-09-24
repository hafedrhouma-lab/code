import time
from transformers import Trainer
from datasets import Dataset
import torch
import mlflow
import numpy as np
import copy
from transformers import TrainingArguments
from torch.quantization import quantize_dynamic

from projects.vendor_ranking.session_based_ranker.utils.preprocess import (
    data_collator_refactored,
    preprocess_function,
    preprocess_test_input,
    is_on_device
)
from projects.vendor_ranking.session_based_ranker.utils.training_utils import (
    get_tokenized_features,
    create_training_args
)
from projects.vendor_ranking.session_based_ranker.utils.tokenizer import (
        create_tokenizer,
        create_fast_tokenizer,
        create_search_tokenizer,
        create_search_fast_tokenizer
    )
from projects.vendor_ranking.session_based_ranker.utils.callbacks import (
        EvaluateEveryNStepsCallbackBucketized,
        EvaluateTrainEveryNStepsCallbackBucketized
    )
from projects.vendor_ranking.session_based_ranker.utils.eval_utils import EvalModel
from projects.vendor_ranking.session_based_ranker.models.t5_recommender import RecommenderClass
from projects.vendor_ranking.session_based_ranker.utils.data_utils import (
        train_val_split_data,
        read_test_data_fs,
        read_test_data_gcs_fs,
        clean_test_data
    )

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    chain_id_tokenizer_columns_offline, max_chain_id_length_offline = get_tokenized_features(params['tokenized_features_offline'], tokenizer_type='chain_id')
    text_tokenizer_columns_offline, max_text_length_offline = get_tokenized_features(params['tokenized_features_offline'], tokenizer_type='text')

    chain_id_tokenizer_columns_online, max_chain_id_length_online = get_tokenized_features(params['tokenized_features_online'], tokenizer_type='chain_id')
    text_tokenizer_columns_online, max_text_length_online = get_tokenized_features(params['tokenized_features_online'], tokenizer_type='text')

    chain_id_tokenizer_columns = list(set(chain_id_tokenizer_columns_offline + chain_id_tokenizer_columns_online))
    text_tokenizer_columns = text_tokenizer_columns_offline + text_tokenizer_columns_online
    max_chain_id_length = max(max_chain_id_length_offline, max_chain_id_length_online)
    max_text_length = max(max_text_length_offline, max_text_length_online)

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
    t5_config_online_chain_id_dict = {
        "vocab_size": fast_chain_id_tokenizer.vocab_size,
        "d_model": params['t5_config_online_chain_id']['d_model'],
        "d_ff": params['t5_config_online_chain_id']['d_ff'],
        "num_layers": params['t5_config_online_chain_id']['num_layers'],
        "num_heads": params['t5_config_online_chain_id']['num_heads'],
        "dropout_rate": params['t5_config_online_chain_id']['dropout_rate'],
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
    t5_config_online_dicts = {
        "chain_id": t5_config_online_chain_id_dict
    }

    # Build dynamic feature configuration based on config file
    feature_configs = []
    for feature in params['tokenized_features_offline']:
        feature_config = {
            "name": feature['name'],
            "tokenizer": fast_tokenizers[feature['tokenizer']],
            "max_length": feature['max_length'],
            "model_type": feature['model_type'],
            "t5_config": t5_config_dicts[feature['tokenizer']]
        }
        feature_configs.append(feature_config)

    online_feature_configs = []
    for feature in params['tokenized_features_online']:
        feature_config = {
            "name": feature['name'],
            "tokenizer": fast_tokenizers[feature['tokenizer']],
            "max_length": feature['max_length'],
            "model_type": feature['model_type'],
            "t5_config": t5_config_online_dicts[feature['tokenizer']]
        }
        online_feature_configs.append(feature_config)

    combined_feature_configs = feature_configs + online_feature_configs

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


    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x,
                                                                             feature_configs=combined_feature_configs,
                                                                             area_id_to_index=area_id_to_index,
                                                                             geohash_to_index=geohash_to_index,
                                                                             numerical_feat_names=numerical_feature_names,
                                                                             label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True, num_proc=1)
    tokenized_train_val_dataset = train_val_dataset.map(lambda x: preprocess_function(x,
                                                                             feature_configs=combined_feature_configs,
                                                                             area_id_to_index=area_id_to_index,
                                                                             geohash_to_index=geohash_to_index,
                                                                             numerical_feat_names=numerical_feature_names,
                                                                             label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True)
    tokenized_val_dataset = val_dataset.map(lambda x: preprocess_function(x,
                                                                             feature_configs=combined_feature_configs,
                                                                             area_id_to_index=area_id_to_index,
                                                                             geohash_to_index=geohash_to_index,
                                                                             numerical_feat_names=numerical_feature_names,
                                                                             label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True)
    tokenized_test_dataset = test_dataset.map(lambda x: preprocess_function(x,
                                                                             feature_configs=combined_feature_configs,
                                                                             area_id_to_index=area_id_to_index,
                                                                             geohash_to_index=geohash_to_index,
                                                                             numerical_feat_names=numerical_feature_names,
                                                                             label_tokenizer=fast_chain_id_tokenizer),
                                                batched=True)


    input_vocab_sizes = [config['tokenizer'].vocab_size for config in combined_feature_configs]
    output_vocab_size = fast_chain_id_tokenizer.vocab_size
    log_dict["input_vocab_sizes"] = input_vocab_sizes
    log_dict["output_vocab_size"] = output_vocab_size

    # Initialize the model
    model = RecommenderClass(
        feature_configs=feature_configs,
        online_feature_configs=online_feature_configs,
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
        dropout_rate=params['dropout_rate'],
        quantize_model=params['quantize_model'],
        compile_model=params['compile_model']
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
                                                          feature_configs=combined_feature_configs,
                                                          label_tokenizer=fast_chain_id_tokenizer,
                                                          area_id_to_index=area_id_to_index,
                                                          geohash_to_index=geohash_to_index,
                                                          numerical_feat_names=numerical_feature_names
                                                          )
    train_eval_callback = EvaluateTrainEveryNStepsCallbackBucketized(eval_steps=freq_eval,
                                                          feature_configs=combined_feature_configs,
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
        data_collator=lambda features: data_collator_refactored(features, combined_feature_configs),
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

    ### MODEL COMPLETED TRAINING
    
    ### MODEL EVALUATIONS BELOW

    model.eval()
    
    if params['quantize_model']:

        quantized_model = copy.deepcopy(model)

        quantized_model = quantized_model.float()
        quantized_model.to(torch.device("cpu"))
        quantized_model.load_model_to_device(torch.device("cpu"))

        # Quantize the dynamic layers (e.g., Linear layers) to qint8
        quantized_model = quantize_dynamic(
            quantized_model,  # The model to quantize
            {torch.nn.Linear},  # Layers to quantize
            dtype=torch.qint8  # Quantized dtype
        )

        eval_model_instance = EvalModel(test_batch_size=params['test_batch_size'],
                                                                              feature_configs=combined_feature_configs,
                                                                              label_tokenizer=fast_chain_id_tokenizer,
                                                                              area_id_to_index=area_id_to_index,
                                                                              geohash_to_index=geohash_to_index,
                                                                              numerical_feat_names=numerical_feature_names,
                                                                              device="cpu"
                                                                              )

        test_metrics = eval_model_instance.eval_model_data(quantized_model, tokenized_test_dataset)

    else:

        eval_model_instance = EvalModel(test_batch_size=params['test_batch_size'],
                                                                              feature_configs=combined_feature_configs,
                                                                              label_tokenizer=fast_chain_id_tokenizer,
                                                                              area_id_to_index=area_id_to_index,
                                                                              geohash_to_index=geohash_to_index,
                                                                              numerical_feat_names=numerical_feature_names
                                                                              )
    
        test_metrics = eval_model_instance.eval_model_data(model, tokenized_test_dataset)

    print("Test metrics after training:")
    print(test_metrics)

    
    ######### Training and Evaluation Complete


    # inference data
    test_data['session_clicks_count'] = test_data['session_clicks'].str.split().str.len()
    filtered_data = test_data[test_data['session_clicks_count'] == 10]
    n = model.offline_model.get_total_hidden_size()
    float32_array = np.random.rand(n).astype(np.float32)
    offline_embedding = float32_array.tolist()
    session_clicks_test = filtered_data.sample(1)["session_clicks"].values.tolist()


    #####################################
    # 1. Cuda Inference
    #####################################
    device = torch.device('cuda')
    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20],
        'session_clicks': session_clicks_test
    }
    model.load_model_to_device(device)
    model.to(device)
    model.eval()

    if model.confirm_model_on_device(device):
        print("all model components on device:", device)
    else:
        print("NOT all model components on device:", device)

    print("model device:", device)
    model_input = preprocess_test_input(
        model_input,
        online_feature_configs,
        device
    )

    if is_on_device(model_input, device):
        print("model_input on device:", device)
    else:
        print("model_input NOT on device:", device)

    test_iterations = 10
    for i in range(test_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, predicted_logits = model.online_model(
                model_input['offline_embedding'],
                model_input['order_hour'],
                model_input['delivery_area_id'],
                model_input['geohash6'],
                labels=None,
                **model_input['tokenized_inputs']
            )
        time_taken = time.time() - start_time
        print(f"Time taken on device {device} : {time_taken}")
        mlflow.log_metric(f"inference_time_{device}_{i}", time_taken)

    #####################################
    # 2. Cpu Inference
    #####################################
    device = torch.device('cpu')
    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20],
        'session_clicks': session_clicks_test
    }
    model.load_model_to_device(device)
    model.to(device)
    model.eval()

    if model.confirm_model_on_device(device):
        print("all model components on device:", device)
    else:
        print("NOT all model components on device:", device)

    print("model device:", device)
    
    model_input = preprocess_test_input(
        model_input,
        online_feature_configs,
        device
    )
        
    if is_on_device(model_input, device):
        print("model_input on device:", device)
    else:
        print("model_input NOT on device:", device)
    
    test_iterations = 10
    for i in range(test_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, predicted_logits = model.online_model(
                model_input['offline_embedding'],
                model_input['order_hour'],
                model_input['delivery_area_id'],
                model_input['geohash6'],
                labels=None,
                **model_input['tokenized_inputs']
            )
        time_taken = time.time() - start_time
        print(f"Time taken on device {device} : {time_taken}")
        mlflow.log_metric(f"inference_time_{device}_{i}", time_taken)

    #####################################
    # 3. Cpu Compiled Inference
    #####################################
    device = torch.device('cpu')
    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20],
        'session_clicks': session_clicks_test
    }
    model.load_model_to_device(device)
    model.to(device)

    compiled_model = copy.deepcopy(model)

    compiled_model.to(torch.device("cpu"))
    compiled_model.load_model_to_device(torch.device("cpu"))

    # Compile Model
    compiled_model.online_model_compiled = torch.compile(compiled_model.online_model, backend="inductor")
    compiled_model.eval()

    if compiled_model.confirm_model_on_device(device):
        print("all model components on device:", device)
    else:
        print("NOT all model components on device:", device)

    print("model device:", device)

    model_input = preprocess_test_input(
        model_input,
        online_feature_configs,
        device
    )

    if is_on_device(model_input, device):
        print("model_input on device:", device)
    else:
        print("model_input NOT on device:", device)

    test_iterations = 10
    for i in range(test_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, predicted_logits = compiled_model.online_model_compiled(
                model_input['offline_embedding'],
                model_input['order_hour'],
                model_input['delivery_area_id'],
                model_input['geohash6'],
                labels=None,
                **model_input['tokenized_inputs']
            )
        time_taken = time.time() - start_time
        print(f"Time taken on device {device} compiled: {time_taken}")
        mlflow.log_metric(f"inference_time_{device}_compiled_{i}", time_taken)

    #####################################
    # 4. Cpu Quantized Inference
    #####################################
    device = torch.device('cpu')
    model_input = {
        'offline_embedding': offline_embedding,
        'order_hour': [20],
        'delivery_area_id': [20],
        'geohash6': [20],
        'session_clicks': session_clicks_test
    }
    model.load_model_to_device(device)
    model.to(device)

    quantized_model = copy.deepcopy(model)

    quantized_model = quantized_model.float()
    quantized_model.to(torch.device("cpu"))
    quantized_model.load_model_to_device(torch.device("cpu"))

    # Quantize the dynamic layers (e.g., Linear layers) to qint8
    quantized_model = quantize_dynamic(
        quantized_model,  # The model to quantize
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8  # Quantized dtype
    )
    quantized_model.eval()

    if quantized_model.confirm_model_on_device(device):
        print("all model components on device:", device)
    else:
        print("NOT all model components on device:", device)

    print("model device:", device)

    model_input = preprocess_test_input(
        model_input,
        online_feature_configs,
        device
    )

    if is_on_device(model_input, device):
        print("model_input on device:", device)
    else:
        print("model_input NOT on device:", device)

    test_iterations = 10
    for i in range(test_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, predicted_logits = quantized_model.online_model(
                model_input['offline_embedding'],
                model_input['order_hour'],
                model_input['delivery_area_id'],
                model_input['geohash6'],
                labels=None,
                **model_input['tokenized_inputs']
            )
        time_taken = time.time() - start_time
        print(f"Time taken on device {device} quantized: {time_taken}")
        mlflow.log_metric(f"inference_time_{device}_quantized_{i}", time_taken)


    recall_10 = round(test_metrics['Test_Recall_10'], 4)
    print(f"Recall at 10 {recall_10}")

    return model, fast_chain_id_tokenizer, recall_10
