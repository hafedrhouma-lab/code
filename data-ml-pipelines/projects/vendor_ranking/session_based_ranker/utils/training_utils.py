from transformers import TrainingArguments

def get_tokenized_features(features, tokenizer_type):
    columns = []
    max_length = 0

    for feature in features:
        if feature['tokenizer'] == tokenizer_type:
            columns.append(feature['name'])
            max_length = max(max_length, feature['max_length'])

    if tokenizer_type=='chain_id':
        columns.append('chain_id')

    return columns, max_length


def create_training_args(training_args_dict):

    training_args = TrainingArguments(
        output_dir="./results",
        save_total_limit=training_args_dict['save_total_limit'],
        save_steps=training_args_dict['save_steps'],
        evaluation_strategy=training_args_dict['evaluation_strategy'],
        learning_rate=training_args_dict['learning_rate'],
        per_device_train_batch_size=training_args_dict['per_device_train_batch_size'],
        per_device_eval_batch_size=training_args_dict['per_device_eval_batch_size'],
        num_train_epochs=training_args_dict['num_train_epochs'],
        weight_decay=training_args_dict['weight_decay'],
        dataloader_num_workers=training_args_dict['dataloader_num_workers'],
        logging_steps=training_args_dict['logging_steps'],
        eval_steps=training_args_dict['eval_steps'],
        gradient_accumulation_steps=training_args_dict['gradient_accumulation_steps'],
        fp16=training_args_dict['fp16'],
        warmup_steps=training_args_dict['warmup_steps'],
        remove_unused_columns=training_args_dict['remove_unused_columns'],
    )

    return training_args