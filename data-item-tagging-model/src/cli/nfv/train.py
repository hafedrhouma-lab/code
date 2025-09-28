import fire
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from ast import literal_eval

from src.model.tokenizer import tokenize, tokenizer
from src.model.metrics import compute_metrics
from src.model.dataset import CustomDataset

from src.utils.log import Logger

from src.data.processor.nfv.tag_mapping import expected_tag_list


class TrainModel:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def train(self, execution_date):
        data = pd.read_csv('src/data/datasets/nfv/cleaned_data.csv')
        data['dummy_vector'] = data['dummy_vector'].apply(literal_eval)

        X = data['cleaned_text']
        y = data['dummy_vector']

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.06, random_state=42
        )

        input_ids_train, attention_masks_train, labels_train = tokenize(x_train, y_train)
        input_ids_test, attention_masks_test, labels_test = tokenize(x_test, y_test)

        labels_train = torch.from_numpy(labels_train)
        labels_test = torch.from_numpy(labels_test)

        id2label = {i: tag for i, tag in enumerate(expected_tag_list)}
        label2id = {tag: i for i, tag in enumerate(expected_tag_list)}

        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            problem_type="multi_label_classification",
            num_labels=len(expected_tag_list),
            id2label=id2label,
            label2id=label2id
        )

        model.to("cpu")
        batch_size = 96
        metric_name = "f1"

        args = TrainingArguments(
            f"roberta-finetuned-nfv-model",  # Specify a RoBERTa model name
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name
        )

        train_dataset = CustomDataset(input_ids_train, attention_masks_train, labels_train)
        val_dataset = CustomDataset(input_ids_test, attention_masks_test, labels_test)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()


if __name__ == "__main__":
    fire.Fire(TrainModel)
