from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os
from collections import Counter


def create_tokenizer(data, vocab_size, tokenizer_filename, columns):
    assert len(columns) > 0
    texts = sum([data[col].astype('str').tolist() for col in columns], [])

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=vocab_size + 5, min_frequency=2,
                               special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(texts, trainer)

    # Save and load the tokenizer using PreTrainedTokenizerFast
    tokenizer.save(tokenizer_filename)


def create_fast_tokenizer(tokenizer_filename, max_length=10):
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename, max_length=max_length,
                                             truncation=True, padding='max_length')
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return fast_tokenizer


def create_search_tokenizer(data, tokenizer_filename, column):
    # Convert column to a string and replace spaces with commas
    data[column] = data[column].str.replace(' ', ', ', regex=False)

    # Prepare the text data for training the tokenizer
    search_texts = data[column].tolist()
    search_text_data = [sentence for text in search_texts for sentence in text.split(', ')]

    # Count unique tokens to determine the vocab size
    tokens = [word for sentence in search_text_data for word in sentence.split()]
    token_counts = Counter(tokens)
    vocab_size = len(token_counts)

    # Create a custom tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train_from_iterator(search_text_data, trainer)

    # Save the tokenizer
    tokenizer.save(tokenizer_filename)


def create_search_fast_tokenizer(tokenizer_filename, max_length=10):
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_filename,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return fast_tokenizer


# Load the feature-specific tokenizers
def load_feature_tokenizers(feature_configs, tokenizer_path):
    for feature_config in feature_configs:
        tokenizer_file = os.path.join(tokenizer_path, f"{feature_config['name']}_tokenizer")
        feature_config['tokenizer'] = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
    return feature_configs