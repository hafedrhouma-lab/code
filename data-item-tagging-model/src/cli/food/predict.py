import fire
import os
from transformers import AutoModelForSequenceClassification

from src.data.processor.utils import remove_noise
from src.model.tokenizer import tokenizer

from src.utils.log import Logger

current_directory = os.path.abspath(__file__)

# Create the path using os.path.join
model_name = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_directory)))),
    "roberta-finetuned-food-model",
    "checkpoint-29710"
)


class PredictTags:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def predict(self, text):
        text = remove_noise(text)

        id2label = {0: 'non_vegetarian', 1: 'vegetarian'}
        label2id = {'non_vegetarian': 0, 'vegetarian': 1}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            problem_type="single_label_classification",
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )

        pred = self.predict_tags(text, model)
        tag = id2label[pred]
        return tag

    @staticmethod
    def predict_tags(text, model, threshold = 0.65):
        import torch.nn.functional as F
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
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = int(probabilities[0, 1] > threshold)

        return predicted_labels


if __name__ == "__main__":
    fire.Fire(PredictTags)
