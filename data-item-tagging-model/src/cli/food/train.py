import fire
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

from src.data.processor.utils import remove_noise
from src.model.tokenizer import tokenize, tokenizer

from src.model.dataset import CustomDataset
from src.model.binary_metrics import compute_metrics
from src.utils.log import Logger


class TrainModel:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def train(self, execution_date):
        data1 = pd.read_csv('src/data/datasets/food/dataset1.csv')
        data2 = pd.read_csv('src/data/datasets/food/dataset2.csv')

        data1 = data1[['item_name', 'item_description', 'label']]
        data2 = data2[['item_name', 'item_description', 'label']]

        data = pd.concat([data1, data2], axis=0)
        data['item_name'] = data['item_name'].astype(str)
        data['item_description'] = data['item_description'].astype(str)

        data['combined'] = data['item_name'] + ' ' + data['item_description']
        data['cleaned_text'] = data['combined'].apply(remove_noise)
        data['label'] = data['label'].apply(lambda x: 0 if x == 'non-vegetarian' else 1)

        #data_temp = data.sample(50)
        X = data['cleaned_text']
        y = data['label']

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.06, random_state=42
        )

        input_ids_train, attention_masks_train, labels_train = tokenize(x_train, y_train)
        input_ids_test, attention_masks_test, labels_test = tokenize(x_test, y_test)

        labels_train = torch.from_numpy(labels_train)
        labels_test = torch.from_numpy(labels_test)

        id2label = {0: 'non-vegetarian', 1: 'vegetarian'}
        label2id = {'non-vegetarian': 0, 'vegetarian': 1}

        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            problem_type="single_label_classification",
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )

        model.to("cpu")
        batch_size = 96

        args = TrainingArguments(
            f"roberta-finetuned-food-model",  # Specify a name for your model
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=True
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
