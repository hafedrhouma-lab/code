import fire
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from src.data.processor.utils import remove_noise
from src.model.tokenizer import tokenizer
from src.data.processor.nfv.tag_mapping import expected_tag_list

from src.utils.log import Logger

current_directory = os.path.abspath(__file__)

# Create the path using os.path.join
model_name = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_directory)))),
    "roberta-finetuned-nfv-model",
    "checkpoint-26190"
)


class PredictTags:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def predict(self, text):
        text = remove_noise(text)

        id2label = {i: tag for i, tag in enumerate(expected_tag_list)}
        label2id = {tag: i for i, tag in enumerate(expected_tag_list)}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            problem_type="multi_label_classification",
            num_labels=len(expected_tag_list),
            id2label=id2label,
            label2id=label2id
        )

        pred = self.predict_tags(text, model)
        predicted_tags = [id2label[idx] for idx, label in enumerate(pred) if label == 1.0]

        return predicted_tags

    @staticmethod
    def predict_tags(text, model, threshold=0.7):
        encoded_input = tokenizer(
            text,
            max_length=60,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = encoded_input["input_ids"].to(model.device)
        attention_mask = encoded_input["attention_mask"].to(model.device)

        # Get predictions from the model
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1

        return predictions


if __name__ == "__main__":
    fire.Fire(PredictTags)
