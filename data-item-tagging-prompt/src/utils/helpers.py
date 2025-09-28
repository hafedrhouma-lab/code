import os
import json
import shutil
from google.cloud import storage


def read_file(path: str):
    with open(path, 'r') as file:
        system_message = file.read()
    return system_message


def save_jsonl_to_gcp(data: str, bucket_name: str, file_path: str):
    """
    Save data as JSONL file to Google Cloud Storage.

    :param data: JSONL-formatted string
    :param bucket_name: Name of the GCS bucket
    :param file_path: Name of the file to save as (include ".jsonl" extension)
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)

    blob.upload_from_string(data)


def read_jsonl_from_gcp(bucket_name: str, file_path: str) -> str:
    """
    Read JSONL data from Google Cloud Storage.

    :param bucket_name: Name of the GCS bucket
    :param file_path: Path to the JSONL file in the GCS bucket (include ".jsonl" extension)
    :return: JSONL data as a string
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)

    jsonl_data = blob.download_as_text()
    return jsonl_data


def save_jsonl_locally(jsonl_data: str, local_file_path: str):
    """
    Save JSONL data to a local file.

    :param jsonl_data: JSONL data as a string
    :param local_file_path: Path to the local file (include ".jsonl" extension)
    """

    if os.path.exists(local_file_path):
        os.remove(local_file_path)

    with open(local_file_path, "w") as f:
        f.write(jsonl_data)


def count_lines_in_jsonl(local_file_path: str) -> int:
    """
    Count the number of lines in a locally saved JSONL file.

    :param local_file_path: Path to the local JSONL file
    :return: Number of lines in the file
    """
    line_count = 0
    with open(local_file_path, "r") as f:
        for _ in f:
            line_count += 1
    return line_count


def read_jsonl_and_get_data(jsonl_local_file_path: str) -> str:
    """
    Read a JSONL file and return its contents as a formatted string.

    :param jsonl_file_path: Path to the JSONL file
    :return: Formatted string containing JSON data
    """
    data_to_write = ""
    with open(jsonl_local_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            try:
                json_object = json.loads(line)
                json_string = json.dumps(json_object)
                data_to_write += json_string + "\n"
            except Exception as e:
                print(f"Error processing line: {e}")

    return data_to_write


def delete_directory(cls, directory_path):
    try:
        shutil.rmtree(directory_path)
        cls._logger.info(f"Directory '{directory_path}' deleted successfully.")
    except OSError as e:
        cls._logger.error(f"Error: {e}")