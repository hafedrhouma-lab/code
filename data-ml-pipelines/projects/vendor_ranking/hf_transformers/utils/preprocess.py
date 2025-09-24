import torch
from torch.utils.data import Dataset as TorchDataset


def data_collator_refactored(features, feature_configs):
    collated_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        collated_features[f'{name}_ids'] = torch.stack(
            [torch.tensor(f[f'{name}_ids']) for f in features])
        collated_features[f'{name}_mask'] = torch.stack(
            [torch.tensor(f[f'{name}_mask']) for f in features])

    collated_features['order_hour'] = torch.tensor([f['order_hour'] for f in features])
    collated_features['delivery_area_id'] = torch.tensor([f['delivery_area_id'] for f in features])
    collated_features['geohash6'] = torch.tensor([f['geohash6'] for f in features])
    collated_features['labels'] = torch.stack([torch.tensor(f['labels']) for f in features])
    collated_features['numerical_features'] = torch.stack([torch.tensor(f['numerical_features']) for f in features])

    return collated_features

def data_collator_without_label(features, feature_configs):
    collated_features = {key: [f[key] for f in features] for key in features[0].keys()}
    # collated_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        collated_features[f'{name}_ids'] = torch.stack(
            [torch.tensor(f[f'{name}_ids']) for f in features])
        collated_features[f'{name}_mask'] = torch.stack(
            [torch.tensor(f[f'{name}_mask']) for f in features])

    collated_features['order_hour'] = torch.tensor([f['order_hour'] for f in features])
    collated_features['delivery_area_id'] = torch.tensor([f['delivery_area_id'] for f in features])
    collated_features['geohash6'] = torch.tensor([f['geohash6'] for f in features])
    collated_features['numerical_features'] = torch.stack([torch.tensor(f['numerical_features']) for f in features])

    return collated_features


def preprocess_function_refactored(examples, feature_configs, area_id_to_index, geohash_to_index, numerical_feat_names, label_tokenizer):
    processed_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        tokenizer = feature_config['tokenizer']
        max_length = feature_config['max_length']

        inputs = tokenizer(examples[name], max_length=max_length, truncation=True, padding="max_length",
                           return_tensors="pt")
        processed_features[f'{name}_ids'] = inputs['input_ids']
        processed_features[f'{name}_mask'] = inputs['attention_mask']

    # Use the separate label tokenizer for labels
    labels = \
    label_tokenizer(examples['chain_id'], max_length=1, truncation=True, padding="max_length", return_tensors="pt")[
        'input_ids']
    processed_features['labels'] = labels

    processed_features['order_hour'] = torch.tensor(examples['order_hour'], dtype=torch.long)
    processed_features['delivery_area_id'] = torch.tensor(
        [area_id_to_index.get(area_id, area_id_to_index['unk']) for area_id in examples['delivery_area_id']],
        dtype=torch.long)
    processed_features['geohash6'] = torch.tensor(
        [geohash_to_index.get(geohash_id, geohash_to_index['unk']) for geohash_id in examples['geohash6']],
        dtype=torch.long)
    processed_features['numerical_features'] = torch.tensor([examples[feat] for feat in numerical_feat_names], dtype=torch.float).T

    return processed_features


def preprocess_evaluation(examples, feature_configs, area_id_to_index, geohash_to_index, numerical_feat_names):
    processed_features = {}

    for feature_config in feature_configs:
        name = feature_config['name']
        tokenizer = feature_config['tokenizer']
        max_length = feature_config['max_length']

        inputs = tokenizer(examples[name], max_length=max_length, truncation=True, padding="max_length",
                           return_tensors="pt").to('cpu')
        processed_features[f'{name}_ids'] = inputs['input_ids']
        processed_features[f'{name}_mask'] = inputs['attention_mask']

    processed_features['order_hour'] = torch.tensor(examples['order_hour'], dtype=torch.long)
    processed_features['delivery_area_id'] = torch.tensor(
        [area_id_to_index.get(area_id, area_id_to_index['unk']) for area_id in examples['delivery_area_id']],
        dtype=torch.long)
    processed_features['geohash6'] = torch.tensor(
        [geohash_to_index.get(geohash_id, geohash_to_index['unk']) for geohash_id in examples['geohash6']],
        dtype=torch.long)
    processed_features['numerical_features'] = torch.tensor([examples[feat] for feat in numerical_feat_names], dtype=torch.float).T

    return processed_features


def preprocess_test_input(model_input, device):
    model_input['offline_embedding'] = torch.tensor([model_input['offline_embedding']], device=device)
    model_input['order_hour'] = torch.tensor([model_input['order_hour']], dtype=torch.long).to(device)
    model_input['delivery_area_id'] = torch.tensor([model_input['delivery_area_id']],
                                                   dtype=torch.long).to(device)
    model_input['geohash6'] = torch.tensor([model_input['geohash6']],
                                                   dtype=torch.long).to(device)

    return model_input


class PreprocessDataset(TorchDataset):
    def __init__(self, samples, feature_configs, numerical_feature_names):
        self.samples = samples
        self.feature_configs = feature_configs
        self.numerical_feature_names = numerical_feature_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        processed_features = {}
        for feature_config in self.feature_configs:
            name = feature_config['name']
            tokenizer = feature_config['tokenizer']
            max_length = feature_config['max_length']

            inputs = tokenizer(sample[name], max_length=max_length, truncation=True, padding="max_length",
                               return_tensors="pt")
            processed_features[f'{name}_ids'] = inputs['input_ids'].squeeze(0)
            processed_features[f'{name}_mask'] = inputs['attention_mask'].squeeze(0)

        processed_features['numerical_features'] = torch.tensor([sample[feat] for feat in self.numerical_feature_names],
                                                                dtype=torch.float)

        return processed_features
