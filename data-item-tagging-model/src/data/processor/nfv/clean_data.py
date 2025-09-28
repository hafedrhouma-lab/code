from src.data.processor.nfv.tag_mapping import tag_mapping, expected_tag_list


def clean_tags(tags):
    if isinstance(tags, str):
        cleaned_tags = tags.replace("'", "").replace("[", "").replace("]", "")
        return cleaned_tags
    else:
        return tags


def clean_tags_list(tag_list):
    cleaned_tags = []
    for tag in tag_list:
        cleaned_tag = tag_mapping.get(tag, tag)
        if cleaned_tag != '' and cleaned_tag in expected_tag_list:
            cleaned_tags.append(cleaned_tag)
    return cleaned_tags