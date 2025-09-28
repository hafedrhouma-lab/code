import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy English language model with parser and NER disabled
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Additional custom stopwords
CUSTOM_STOPWORDS = frozenset([
    'ml', 'g', 'kg', 'pet', 'gm', 'x', 'cm', 'mm', 'pcs', 'pack', 'packs', 'packaging',
    'size', 'flavor', 'color', 'brand', 'product', 'description', 'contains', 'ingredients',
    'weight', 'piece', 'pieces', 'item', 'items', 'package', 'type', 'branding', 'volume',
    'content', 'quantity', 'container', 'form', 'available', 'varieties', 'variant', 'variants',
    'code', 'stock', 'number', 'features', 'feature', 'specifications', 'specification', 'specify',
    'carton', 'name', 'liters', 'grams', 'kilograms', 'milliliters', 'centimeters', 'quantity', 'size',
    'weights', 'packed', 'packing', 'pieces', 'piece', 'numbers', 'products', 'items', 'brands',
    'descriptions', 'types', 'availability', 'cartons'
])

STOPWORDS = STOP_WORDS.union(CUSTOM_STOPWORDS)


def remove_noise(text):
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove words with one or two characters
    text = ' '.join([word if len(word) > 2 else ' ' for word in text.split()])

    # Remove stopwords
    text = ' '.join([word if word not in STOPWORDS else ' ' for word in text.split()])

    # Remove specific patterns (e.g., '2x500ml', '3X500G', '48x170GM', '2x900g', '500g', '3X900g', '2x399g')
    text = re.sub(r'\b\d+[xX]\d+(ml|g|kg|gm)\b', ' ', text)
    text = re.sub(r'\b\d+(ml|g|kg|gm)\b', ' ', text)

    # Remove digits and specific patterns (e.g., '2kg')
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\b\d+\s*(?:k(?:g|g?|ilo(?:s)?)?|g(?:ram(?:s)?)?|ml|millilit(?:er|re)s?)\b', ' ', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text


def create_dummy_vector(tag_list, tags):
    dummy_vector = [1 if tag in tags else 0 for tag in tag_list]
    return dummy_vector
