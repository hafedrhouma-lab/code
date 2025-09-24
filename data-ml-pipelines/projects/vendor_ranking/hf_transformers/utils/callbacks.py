from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from transformers import EarlyStoppingCallback

import mlflow


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=3, improvement_threshold=0.5, greater_is_better=True):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.improvement_threshold = improvement_threshold
        self.best_metric = None  # Initialize to store the best metric
        self.greater_is_better = greater_is_better

    def on_evaluate(self, args, state, control, **kwargs):
        metric_value = state.best_metric
        if metric_value is not None:
            # Logic to save the best model and track early stopping
            if self.best_metric is None or \
                    (self.greater_is_better and metric_value > self.best_metric + self.improvement_threshold) or \
                    (not self.greater_is_better and metric_value < self.best_metric - self.improvement_threshold):
                self.best_metric = metric_value
                self.early_stopping_patience_counter = 0  # Reset patience counter

                state.best_model_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                state.best_model_checkpoint = checkpoint_path
                print(f"Updating best model checkpoint to: {checkpoint_path}")
                control.should_save = True  # Force saving of the new best model

                print("Early Stopping Not Needed")
            else:
                self.early_stopping_patience_counter += 1
                print("Wait Count + 1")
                print(f"early_stopping_patience_counter: {self.early_stopping_patience_counter}")
                print(f"early_stopping_patience: {self.early_stopping_patience}")
                if self.early_stopping_patience_counter >= self.early_stopping_patience:
                    control.should_training_stop = True
                    print("Early Stopping")


class EvaluateEveryNStepsCallbackBucketized(TrainerCallback):
    def __init__(self, eval_steps, batch_size=1024, feature_configs=None, label_tokenizer=None,
                 area_id_to_index=None, geohash_to_index=None, numerical_feat_names=None):
        self.eval_steps = eval_steps
        self.batch_size = batch_size  # Batch size for evaluation
        self.K_values = [5, 10, 20]
        self.feature_configs = feature_configs
        self.label_tokenizer = label_tokenizer
        self.fast_tokenizer = label_tokenizer
        self.area_id_to_index = area_id_to_index
        self.geohash_to_index = geohash_to_index
        self.numerical_feat_names = numerical_feat_names
        self.metrics = {}  # To store the metrics


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.eval_steps == 0) or (state.global_step == 1):
            control.should_evaluate = True
            self.metrics = self.evaluate(args, state, control)

            # if "eval_recall_at_10" in self.metrics:
            #     state.best_metric = self.metrics["eval_recall_at_10"]
            #     self.trainer.state.best_metric = self.metrics["eval_recall_at_10"]
            # else:
            #     print("Warning: 'eval_recall_at_10' not found in metrics!")

    def evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        model = self.trainer.model
        eval_dataset = self.trainer.eval_dataset
        model.eval()
        # self.run_evaluation_iteration(eval_dataset, model, args.device)
        # Use DataLoader for batch processing
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)
        with torch.no_grad():
            metrics = self.run_evaluation_iteration(eval_dataloader, model, args.device, state)

        return metrics

    def process_batch(self, batch, device):
        processed_features = {}

        # Iterate through feature configs to handle tokenized features
        for feature_config in self.feature_configs:
            name = feature_config['name']
            tokenizer = feature_config['tokenizer']
            max_length = feature_config['max_length']

            inputs = tokenizer(batch[name], max_length=max_length, truncation=True, padding="max_length",
                                         return_tensors="pt").to(device)
            processed_features[f'{name}_ids'] = inputs['input_ids']
            processed_features[f'{name}_mask'] = inputs['attention_mask']

        # Process labels
        processed_features['labels'] = self.label_tokenizer(batch['chain_id'], max_length=1, truncation=True,
                                                          padding="max_length", return_tensors="pt")['input_ids'].to(device)
        processed_features['order_hour'] = batch['order_hour'].clone().detach().long().to(device)
        processed_features['delivery_area_id'] = batch['delivery_area_id'].to(device)
        processed_features['geohash6'] = batch['geohash6'].to(device)
        # processed_features['delivery_area_id'] = torch.tensor(
        #     [self.area_id_to_index[area_id] if area_id in self.area_id_to_index else 0 for area_id in batch['delivery_area_id']],
        #     dtype=torch.long).to(device)
        # processed_features['geohash'] = torch.tensor(
        #     [self.geohash_to_index[geohash_id] if geohash_id in self.geohash_to_index else 0 for geohash_id in batch['geohash']],
        #     dtype=torch.long).to(device)

        numerical_feat_tensors = [batch[feat].clone().detach().float() for feat in self.numerical_feat_names]
        processed_features['numerical_features'] = torch.stack(numerical_feat_tensors, dim=1).to(device)
        return processed_features

    def run_evaluation_iteration(self, eval_dataloader, model, device, state):

        success_count_list = [0] * len(self.K_values)
        total_count = 0
        buckets = {
            "reorder": {"success": [0] * len(self.K_values), "total": 0},
            "discovery": {"success": [0] * len(self.K_values), "total": 0},
            "reorder_in_prev_chains": {"success": [0] * len(self.K_values), "total": 0},
            "reorder_in_freq_chains": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_in_freq_clicks": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_in_prev_clicks": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_not_in_clicks": {"success": [0] * len(self.K_values), "total": 0},
        }

        for batch in tqdm(eval_dataloader):
            processed_batch = self.process_batch(batch, device)
            loss, logits = model(**processed_batch)

            subset_keys = ['labels', 'prev_chains', 'freq_chains', 'prev_clicks', 'freq_clicks', 'chain_id']

            # Create a new dictionary with the subset of keys
            temp_batch = {key: batch[key] for key in subset_keys}
            temp_batch['labels'] = [torch.tensor([value]) for value in temp_batch['labels'][0]]

            batch_df = pd.DataFrame(temp_batch)

            for i, example in batch_df.iterrows():
                target_token_id = processed_batch['labels'][i][0].item()
                for index, k_value in enumerate(self.K_values):
                    top_K_predictions = torch.topk(logits[i], k_value).indices.tolist()

                    if target_token_id in top_K_predictions:
                        success_count_list[index] += 1

                    self.categorize_row(example, target_token_id, top_K_predictions, index, buckets)

                total_count += 1

        metrics = {}
        for count, k in zip(success_count_list, self.K_values):
            success_rate = (count / total_count) * 100
            print(f"Recall@{k} = {success_rate:.2f}%")
            mlflow.log_metric(f"Val_Recall_{k}", success_rate, step=state.global_step)
            metrics[f"Val_Recall_{k}"] = success_rate
            # metrics[f"eval_recall_at_{k}"] = success_rate

        for bucket, counts in buckets.items():
            if counts["total"] > 0:
                buckets[bucket]["total"] = buckets[bucket]["total"] / (len(self.K_values))

        # Print success rates for each bucket
        for bucket, counts in buckets.items():
            if counts["total"] > 0:
                for count, name in zip(counts["success"], self.K_values):
                    b_success_rate = (count / counts["total"]) * 100
                    b_percentage = (counts["total"] / total_count) * 100
                    print(f"""Val_{bucket.capitalize()} Recall@{name} = {b_success_rate:.2f}% ({b_percentage:.2f}% of total)""")
                    mlflow.log_metric(f"Val_{bucket.capitalize()}_Recall_{name}", b_success_rate,step=state.global_step)
                    mlflow.log_metric(f"Val_{bucket.capitalize()}_Percentage", b_percentage, step=state.global_step)
                    # if (bucket == "discovery") and (name == 10):
                    #     metrics["val_disc_recall_at_10"] = b_success_rate

        return metrics

    def categorize_row(self, example, target_token_id, top_k_predictions, index, buckets):

        prev_chains = set(map(str, example['prev_chains'].split()))
        freq_chains = set(map(str, example['freq_chains'].split()))
        prev_clicks = set(map(str, example['prev_clicks'].split()))
        freq_clicks = set(map(str, example['freq_clicks'].split()))

        is_reorder_in_freq_chains = str(example['chain_id']) in freq_chains
        is_reorder_in_prev_chains = str(example['chain_id']) in prev_chains
        is_reorder = is_reorder_in_freq_chains or is_reorder_in_prev_chains

        is_discovery = not is_reorder
        is_discovery_in_freq_clicks = is_discovery and (str(example['chain_id']) in freq_clicks)
        is_discovery_in_prev_clicks = is_discovery and (str(example['chain_id']) in prev_clicks)
        is_discovery_not_in_clicks = is_discovery and (not is_discovery_in_freq_clicks) and (
            not is_discovery_in_prev_clicks)

        success = target_token_id in top_k_predictions

        if success:
            if is_reorder:
                buckets["reorder"]["success"][index] += 1
                buckets["reorder"]["total"] += 1
            if is_reorder_in_prev_chains:
                buckets["reorder_in_prev_chains"]["success"][index] += 1
                buckets["reorder_in_prev_chains"]["total"] += 1
            if is_reorder_in_freq_chains:
                buckets["reorder_in_freq_chains"]["success"][index] += 1
                buckets["reorder_in_freq_chains"]["total"] += 1
            if is_discovery:
                buckets["discovery"]["success"][index] += 1
                buckets["discovery"]["total"] += 1
            if is_discovery_in_freq_clicks:
                buckets["discovery_in_freq_clicks"]["success"][index] += 1
                buckets["discovery_in_freq_clicks"]["total"] += 1
            if is_discovery_in_prev_clicks:
                buckets["discovery_in_prev_clicks"]["success"][index] += 1
                buckets["discovery_in_prev_clicks"]["total"] += 1
            if is_discovery_not_in_clicks:
                buckets["discovery_not_in_clicks"]["success"][index] += 1
                buckets["discovery_not_in_clicks"]["total"] += 1
        else:
            if is_reorder:
                buckets["reorder"]["total"] += 1
            if is_reorder_in_prev_chains:
                buckets["reorder_in_prev_chains"]["total"] += 1
            if is_reorder_in_freq_chains:
                buckets["reorder_in_freq_chains"]["total"] += 1
            if is_discovery:
                buckets["discovery"]["total"] += 1
            if is_discovery_in_freq_clicks:
                buckets["discovery_in_freq_clicks"]["total"] += 1
            if is_discovery_in_prev_clicks:
                buckets["discovery_in_prev_clicks"]["total"] += 1
            if is_discovery_not_in_clicks:
                buckets["discovery_not_in_clicks"]["total"] += 1


class EvaluateTrainEveryNStepsCallbackBucketized(EvaluateEveryNStepsCallbackBucketized):
    def __init__(self, eval_steps, batch_size=1024, feature_configs=None, label_tokenizer=None,
                 area_id_to_index=None, geohash_to_index=None, numerical_feat_names=None, eval_data=None):

        super().__init__(eval_steps, batch_size, feature_configs, label_tokenizer,
                         area_id_to_index, geohash_to_index, numerical_feat_names)
        self.eval_data = eval_data

    def evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        model = self.trainer.model
        eval_dataset = self.eval_data

        model.eval()
        # Use DataLoader for batch processing
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            self.run_evaluation_iteration(eval_dataloader, model, args.device, state)

    def run_evaluation_iteration(self, eval_dataloader, model, device, state):
        success_count_list = [0] * len(self.K_values)
        total_count = 0
        buckets = {
            "reorder": {"success": [0] * len(self.K_values), "total": 0},
            "discovery": {"success": [0] * len(self.K_values), "total": 0},
            "reorder_in_prev_chains": {"success": [0] * len(self.K_values), "total": 0},
            "reorder_in_freq_chains": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_in_freq_clicks": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_in_prev_clicks": {"success": [0] * len(self.K_values), "total": 0},
            "discovery_not_in_clicks": {"success": [0] * len(self.K_values), "total": 0},
        }

        for batch in tqdm(eval_dataloader):
            processed_batch = self.process_batch(batch, device)
            loss, logits = model(**processed_batch)

            subset_keys = ['labels', 'prev_chains', 'freq_chains', 'prev_clicks', 'freq_clicks', 'chain_id']

            # Create a new dictionary with the subset of keys
            temp_batch = {key: batch[key] for key in subset_keys}
            temp_batch['labels'] = [torch.tensor([value]) for value in temp_batch['labels'][0]]

            batch_df = pd.DataFrame(temp_batch)

            for i, example in batch_df.iterrows():
                target_token_id = processed_batch['labels'][i][0].item()
                for index, k_value in enumerate(self.K_values):
                    top_K_predictions = torch.topk(logits[i], k_value).indices.tolist()

                    if target_token_id in top_K_predictions:
                        success_count_list[index] += 1

                    self.categorize_row(example, target_token_id, top_K_predictions, index, buckets)

                total_count += 1

        for count, name in zip(success_count_list, self.K_values):
            success_rate = (count / total_count) * 100
            print(f"Train_Recall@{name} = {success_rate:.2f}%")
            mlflow.log_metric(f"Train_Recall_{name}", success_rate, step=state.global_step)

        for bucket, counts in buckets.items():
            if counts["total"] > 0:
                buckets[bucket]["total"] = buckets[bucket]["total"] / len(self.K_values)

        # Print success rates for each bucket
        for bucket, counts in buckets.items():
            if counts["total"] > 0:
                for count, name in zip(counts["success"], self.K_values):
                    b_success_rate = (count / counts["total"]) * 100
                    b_percentage = (counts["total"] / total_count) * 100
                    print(
                        f"""Train_{bucket.capitalize()} Recall@{name} = {b_success_rate:.2f}% ({b_percentage:.2f}% of total)""")
                    mlflow.log_metric(f"Train_{bucket.capitalize()}_Recall_{name}", b_success_rate,step=state.global_step)
                    mlflow.log_metric(f"Train_{bucket.capitalize()}_Percentage", b_percentage, step=state.global_step)
