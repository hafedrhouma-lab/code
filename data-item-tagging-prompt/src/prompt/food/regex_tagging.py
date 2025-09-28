non_veg_keywords = [
    "chicken", "keema", "pigeon", "non veg", "goat", "lamb", "salami", "pastrami",
    "kebab", "kabab", "duck", "liver", "seafood", "sea food", "lobster", "prawn", "crab", "fish",
    "meat", "shrimp", "steak", "calamari", "hotdog", "beef", "salmon", "sheesh tawook", "shish tawook",
    "shish tawouk", "murgh", "fillet", "angus", "mutton", "squid", "ribs", "turkey", "zinger",
    "zinker", "sausage", "tuna", "camel", "tilapia", "pepperoni", "shawarma", "shawerma", "hot dog",
    "fish items", "bolognese"
]

veg_keywords = [
    "milkshake", "juice", "acai", "frappe", "beverage", "hot and cold drinks",
    "tea", "drink", "juice bottle", "juices", "shakes and sodas", "softdrink",
    "coffee", "mojito", "sodas and water", "coffee", "milkshake", "shake",
    "frappe", "cocktail", "mocktail", "hot chocolate", "vegetarian",
    "fruits and grains", "kunafa", "dessert", "sweet", "smoothie", "plant based", 'vegan',
    'latte', 'nutella', 'strawberry', 'banana', 'mango', 'water', 'cookie'
]


def get_label_regex(row):
    item_text = f"{row['item_name_en'].lower()} {row['item_description_en'].lower()}"

    # Check for non-vegetarian keywords
    if any(keyword in item_text for keyword in non_veg_keywords):
        return 'non_vegetarian'

    # Check for vegetarian keywords
    if any(keyword in item_text for keyword in veg_keywords):
        return 'vegetarian'

    # No match found
    return 'unknown'