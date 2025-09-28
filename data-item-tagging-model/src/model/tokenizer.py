import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Replace with the desired model


def tokenize(text_input, labels):
    text_list = text_input.tolist()
    input_data = [str(text) for text in text_list]

    labels_train = np.array(labels.tolist(), dtype=int)

    encoded_dict = tokenizer.batch_encode_plus(
        input_data,
        max_length=60,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True,
    )

    input_ids_train = encoded_dict['input_ids']
    attention_masks_train = encoded_dict['attention_mask']

    return input_ids_train, attention_masks_train, labels_train