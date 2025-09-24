import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from transformers import T5Config, TrainingArguments, Trainer
from utils.data_loader import read_data_hs, read_data
from utils.preprocess import preprocess_function_refactored
from utils.tokenizer import create_tokenizer, create_fast_tokenizer, load_feature_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from utils.callbacks import EvaluateEveryNStepsCallbackBucketized
from models.t5_recommender import RecommenderClass
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Data Collator Function
def data_collator_refactored(features, feature_configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    collated_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        collated_features[f'{name}_ids'] = torch.stack([torch.tensor(f[f'{name}_ids']) for f in features]).to(device)
        collated_features[f'{name}_mask'] = torch.stack([torch.tensor(f[f'{name}_mask']) for f in features]).to(device)

    collated_features['order_hour'] = torch.tensor([f['order_hour'] for f in features]).to(device)
    collated_features['delivery_area_id'] = torch.tensor([f['delivery_area_id'] for f in features]).to(device)
    collated_features['labels'] = torch.stack([torch.tensor(f['labels']) for f in features]).to(device)

    return collated_features

# Load data
data = read_data(country_code='EG', data_points=403000)

data['chain_id'] = data['chain_id'].astype('str')
data['order_hour'] = data['order_time_utc'].dt.hour
data['delivery_area_id'] = data['delivery_area_id']


# Prepare tokenizer
word_tokenizer = create_tokenizer(data, vocab_size=data['chain_id'].nunique(),
                                  tokenizer_filename='word_level_tokenizer.json',
                                  columns=['prev_chains', 'freq_chains', 'chain_id'])
fast_tokenizer = create_fast_tokenizer(tokenizer_filename='word_level_tokenizer.json', max_length=10)


# Example configuration with transformer types included
feature_configs = [
    {"name": "sess_clicks", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
    {"name": "prev_chains", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
    {"name": "freq_chains", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
]


# Prepare dataset
unique_area_ids = data['delivery_area_id'].unique()
area_id_to_index = {area_id: index for index, area_id in enumerate(unique_area_ids)}

train = data[:-400000]
eval = data[-400000:].sample(1000)

train_dataset = Dataset.from_pandas(train)
eval_dataset = Dataset.from_pandas(eval)

# Tokenized datasets
tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                     feature_configs=feature_configs,
                                                                                     area_id_to_index=area_id_to_index,
                                                                                     label_tokenizer=fast_tokenizer),
                                            batched=True, num_proc=32)
tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function_refactored(x,
                                                                                   feature_configs=feature_configs,
                                                                                   area_id_to_index=area_id_to_index,
                                                                                   label_tokenizer=fast_tokenizer),
                                          batched=True)

# Get vocab sizes for input features
input_vocab_sizes = [config['tokenizer'].vocab_size for config in feature_configs]

# Define the output vocab size as the label tokenizer's vocab size
output_vocab_size = fast_tokenizer.vocab_size
print('vocab size', output_vocab_size)


# Initialize the model
model = RecommenderClass(
    feature_configs=feature_configs,
    order_hour_dim=16,
    area_id_vocab_size=len(area_id_to_index),
    area_id_dim=16,
    input_vocab_sizes=[fast_tokenizer.vocab_size for _ in feature_configs],
    output_vocab_size=fast_tokenizer.vocab_size,
    area_id_to_index=area_id_to_index
).to(device)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_total_limit=5,
    save_steps=1000,
    evaluation_strategy="epoch",
    save_strategy='no',
    learning_rate=2e-4,
    per_device_train_batch_size=1024,
    per_device_eval_batch_size=200,
    num_train_epochs=1,
    weight_decay=0.01,
    remove_unused_columns=False
)

# Initialize callback
eval_callback = EvaluateEveryNStepsCallbackBucketized(200, fast_tokenizer, area_id_to_index)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=fast_tokenizer,
    data_collator=lambda features: data_collator_refactored(features, feature_configs),
    callbacks=[eval_callback]
)

# Set the trainer instance in the callback
eval_callback.trainer = trainer

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_models("./trained_model")
fast_tokenizer.save_pretrained("./trained_model")


# Define paths
offline_path = "./trained_model"
online_path = "./trained_model"
tokenizer_path = "./trained_model/tokenizers"

# Load the models and tokenizers
feature_configs = [
    {"name": "sess_clicks", "tokenizer": None, "max_length": 10, "model_type": "T5"},
    {"name": "prev_chains", "tokenizer": None, "max_length": 10, "model_type": "T5"},
    {"name": "freq_chains", "tokenizer": None, "max_length": 10, "model_type": "T5"},
]
# Load feature-specific tokenizers
feature_configs = load_feature_tokenizers(feature_configs, tokenizer_path)


# Load the models
offline_model = RecommenderClass.load_offline_model(offline_path)
online_model, loaded_area_id_to_index = RecommenderClass.load_online_model(online_path)
loaded_model = RecommenderClass.load_model(online_path)

# Move models to the appropriate device
offline_model = offline_model.to(device)
online_model = online_model.to(device)
loaded_model = loaded_model.to(device)
model.eval()
offline_model.eval()
online_model.eval()
loaded_model.eval()

# Take a few samples from the eval_dataset
test_samples = eval_dataset.select([0, 1, 2])  # Select the first three samples for testing


# Preprocess the test samples using feature-specific tokenizers
def preprocess_sample(sample, feature_configs, loaded_area_id_to_index):
    processed_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        tokenizer = feature_config['tokenizer']
        max_length = feature_config['max_length']

        inputs = tokenizer(sample[name], max_length=max_length, truncation=True, padding="max_length",
                           return_tensors="pt")
        processed_features[f'{name}_ids'] = inputs['input_ids'].squeeze(0).to(device)
        processed_features[f'{name}_mask'] = inputs['attention_mask'].squeeze(0).to(device)

    labels = feature_configs[0]['tokenizer'](sample['chain_id'], max_length=1, truncation=True, padding="max_length",
                                             return_tensors="pt")['input_ids']
    order_hour = torch.tensor([sample['order_hour']], dtype=torch.long).to(device)
    delivery_area_id = torch.tensor([loaded_area_id_to_index[sample['delivery_area_id']]], dtype=torch.long).to(device)

    processed_features['order_hour'] = order_hour
    processed_features['delivery_area_id'] = delivery_area_id
    processed_features['labels'] = labels.squeeze(0).to(device)

    return processed_features

preprocessed_samples = [preprocess_sample(sample, feature_configs, loaded_area_id_to_index) for sample in test_samples]

# Testing the loaded models
for i, sample in enumerate(preprocessed_samples):

    # Unsqueeze the values in the sample dictionary
    sample = {key: value.unsqueeze(0) for key, value in sample.items()}
    # Forward pass through the original model
    with torch.no_grad():
        _, original_logits = model(**sample)

    ## for streamlit app
    with torch.no_grad():
        _, loaded_e2e_logits = loaded_model(**sample)

    # Forward pass through the loaded offline and online models
    offline_inputs = {key: value for key, value in sample.items() if key.endswith('_ids') or key.endswith('_mask')}

    # Time the offline model
    start_time = time.time()
    with torch.no_grad():
        offline_output = offline_model(**offline_inputs)
    end_time = time.time()
    offline_time = end_time - start_time
    print(f"Offline model execution time: {offline_time:.6f} seconds")

    # Time the online model
    start_time = time.time()
    with torch.no_grad():
        _, loaded_logits = online_model(offline_output, sample['order_hour'], sample['delivery_area_id'],
                                        sample['labels'])
    end_time = time.time()
    online_time = end_time - start_time
    print(f"Online model execution time: {online_time:.6f} seconds")

    # Compare outputs
    if torch.allclose(original_logits, loaded_logits, atol=1e-6) and \
            torch.allclose(original_logits, loaded_e2e_logits, atol=1e-6):
        print(f"Test case {i} passed: The outputs match!")
    else:
        print(f"Test case {i} failed: The outputs do not match.")



