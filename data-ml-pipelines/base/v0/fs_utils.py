import os
import re
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
import pickle
import logging
import json
from pandas import DataFrame
from typing import Any
import yaml


def check_dir(dir_path, mkdir=True):
    """check_dir function checks if a directory exists and creates it if not

    Args:
        dir_path (:obj:`str`): path to the desired directory (including directory name)
        mkdir (:obj:`bool`, optional): Creates the directory if it does not exist. Defaults to True.
    Returns:
        status :obj:`bool`: Whether or not the directory existed
    """
    dir_exist = os.path.exists(dir_path)
    if not dir_exist and mkdir:
        os.makedirs(dir_path)
    tmp_dir_exist = os.path.exists(dir_path)
    if not tmp_dir_exist and mkdir:
        raise ValueError("Failed to create dir '%s'" % dir_path)
    return dir_exist


def findfiles(dirpath=".", fileregex="", regex="", recur=False, gcs=False):
    """Scan a directory for files.

    Args:
        dirpath (:obj:`str`): Optional. Default '.'.
            The directory to scan.
        fileregex (:obj:`str`): Optional. Default ''.
            A regular expression pattern to filter the files.
        regex (:obj:`str`): Optional. Default ''.
            A regular expression pattern to filter results
        recur (:obj:`bool`): Optional. Default False.
            Option to enable search recursively.
        gcs (:obj:`bool`): whether the search should happen on GCS storage
    Returns:
        :obj:`str`: List of files found.

    """
    lst = []
    if gcs:
        file_path_split = dirpath.split("/", 1)
        GS_BUCKET_NAME = file_path_split[0]
        GS_BLOB_NAME = file_path_split[-1]

        client = storage.Client("tlb-data-dev")
        all_files = []
        for blob in client.list_blobs(
            GS_BUCKET_NAME, prefix=GS_BLOB_NAME, delimiter="/"
        ):
            all_files.append(blob.name)
        all_files.remove(GS_BLOB_NAME)
        for path in all_files:
            try:
                f = path.rsplit("/", 1)[-1]
                if fileregex != "":
                    if re.search(fileregex, f):
                        lst.append(path + os.sep + f)
                else:
                    # lst.append(path + os.sep + f)
                    lst.append(path)
            except Exception as e:
                raise ValueError(e)

        lst = [i for i in lst if re.search(regex, i)]
    else:  # local file system
        pathtree = os.walk(dirpath)

        for path, dirs, files in pathtree:
            for f in files:
                try:
                    if fileregex != "":
                        if re.search(fileregex, f):
                            lst.append(path + os.sep + f)
                    else:
                        lst.append(path + os.sep + f)
                except Exception as e:
                    raise ValueError(e)
            if not recur:
                break
        lst = [i for i in lst if re.search(regex, i)]
    return lst


def load_from_file(path_filename: str, **kwargs) -> pd.DataFrame:
    """Your function's docstring"""

    extension = os.path.splitext(path_filename)[
        1
    ].lower()  # Ensure it's lowercase for comparison

    if extension in [".pqt", ".parquet", ".parquet.gzip"]:
        data = pd.read_parquet(path_filename, **kwargs)
    elif extension == ".pkl":
        with open(path_filename, "rb") as openfile:
            try:
                data = pickle.load(openfile)
            except Exception as e:
                raise ValueError(f"Failed to load pickle file. Error: {e}")
    elif extension == ".txt":
        with open(path_filename, encoding="utf-8") as file:
            data = [line.strip() for line in file]
    elif extension == ".npy":
        data = pd.DataFrame(np.load(path_filename, allow_pickle=True), **kwargs).drop(
            columns=["index"]
        )
    elif extension == ".csv":
        data = pd.read_csv(path_filename, **kwargs)
    elif extension == ".gz" and ".csv" in path_filename:
        data = pd.read_csv(path_filename, compression="gzip", **kwargs)
    elif extension == ".tsv":
        data = pd.read_csv(path_filename, sep="\t", **kwargs)
    elif extension == ".zip":
        data = pd.read_csv(path_filename, **kwargs)
    elif extension == ".xlsx":
        data = pd.read_excel(path_filename, **kwargs)
    elif extension == ".json":
        data = pd.read_json(path_filename, **kwargs)
    elif extension in [".yaml", ".yml"]:
        with open(path_filename, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
    else:
        raise ValueError("Unknown extension - data is not loaded!")

    return data


def save_to_file(df, path_filename, **kwargs):
    """Save pandas dataframe to one of the following extensions:
        - \*.pkl
        - \*.npy
        - \*.pqt,
        - \*.parquet
        - \*.csv
        - \*.xlsx

    **note**
        - \*.csv1 saves dtypes into row1 of csv1 file

    Args:
        df (:obj:`pd.DataFrame`): dataframe to be saved
        path_filename (:obj:`str`): path, filename and extension
            for where to save

    """
    pqt = [".pqt", ".parquet", "parquet.gzip"]
    extension = os.path.splitext(path_filename)[1]
    basename = os.path.basename(path_filename)
    if any(list(map(lambda x: x in basename, pqt))):
        table = pa.Table.from_pandas(df, preserve_index=False, **kwargs)
        pq.write_table(table, path_filename)
    elif extension == ".pkl":
        if isinstance(df, pd.DataFrame):
            df.to_pickle(path_filename, **kwargs)
        else:
            import pickle

            with open(path_filename, "wb") as handle:
                pickle.dump(df, handle, **kwargs)
    elif extension == ".npy":
        np.save(path_filename, df.to_records(**kwargs))
    elif extension == ".csv":
        df.to_csv(path_filename, **kwargs)
    elif (extension == ".gz") & (".csv" in path_filename):
        df.to_csv(path_filename, **kwargs)
    elif extension == ".tsv":
        df.to_csv(path_filename, sep="\t", **kwargs)
    elif extension == ".xlsx":
        df.to_excel(path_filename, **kwargs)
    elif extension == ".json":
        import json

        with open(path_filename, "w", encoding="utf8") as json_file:
            json.dump(df.to_json(**kwargs), json_file, ensure_ascii=False)
    else:
        raise ValueError("unknown extension - data not saved")

    return


def load_from_gcs(gcs_path: str, **kwargs) -> Any:
    """
    Downloads a file from the GCS bucket and returns its content using read_from_file.

    Args:
        gcs_path: Full path including bucket name + blob name + filename.

    Returns:
        The content of the file.
    """
    GS_BUCKET_NAME, GS_BLOB_NAME = gcs_path.split("/", 1)
    FILENAME = GS_BLOB_NAME.rsplit("/", 1)[-1]
    file_extension = get_file_extension(FILENAME)

    storage_client = storage.Client("tlb-data-dev")
    bucket = storage_client.get_bucket(GS_BUCKET_NAME)
    blob = bucket.blob(GS_BLOB_NAME)
    data = None

    # Else, load it as binary

    if file_extension == "csv":
        data = pd.read_csv(f"gs://{GS_BUCKET_NAME}/{GS_BLOB_NAME}")
        logging.debug(f"File downloaded directly using pandas")

    elif file_extension == "pkl":
        pickle_in = blob.download_as_string()
        data = pickle.loads(pickle_in)
        logging.info(f"object is downloaded using pickle")

    else:  # General method to save the object locally then load it
        # Temporary file to store downloaded data
        TMP_FILE = f"tmp_{FILENAME}"

        # Download the blob to the temporary file
        blob.download_to_filename(TMP_FILE)

        # Use read_from_file to process the content
        data = load_from_file(TMP_FILE, **kwargs)

        os.remove(TMP_FILE)

    return data


def save_to_gcs(
    obj: Any = None, gcs_path: str = None, local_path: str = None, **kwargs
) -> None:
    """
    Uploads a file to the GCS bucket.

    Args:
        obj: Object to upload on GCS.
        gcs_path: Full path including bucket name + blob name + filename.
        local_path: Local file path to be loaded and then uploaded to GCS.

        Note: if local_path is provided, `obj` will be ignored
    """

    GS_BUCKET_NAME, GS_BLOB_NAME = gcs_path.split("/", 1)
    FILENAME = GS_BLOB_NAME.rsplit("/", 1)[-1]
    file_extension = get_file_extension(FILENAME)

    storage_client = storage.Client("tlb-data-dev")
    bucket = storage_client.get_bucket(GS_BUCKET_NAME)
    blob = bucket.blob(GS_BLOB_NAME)

    # If a local path is provided, load data from it
    if local_path:
        obj = load_from_file(local_path)

    # If kwargs are provided and file_extension matches the list, save locally first
    if kwargs and file_extension in ["pkl", "npy", "pqt", "parquet", "csv", "xlsx"]:
        logging.debug("kwargs passed")
        logging.info(f"Saving and uploading {FILENAME} ...")
        save_to_file(obj, f"TMP_{FILENAME}", **kwargs)
        blob.upload_from_filename(f"TMP_{FILENAME}")
        os.remove(f"TMP_{FILENAME}")
    else:
        if isinstance(obj, DataFrame):
            logging.debug("No passed kwargs, binary upload is used")
            if file_extension == "csv":
                blob.upload_from_string(obj.to_csv(index=False), "text/csv")
            elif file_extension == "json":
                json_str = obj.to_json(**kwargs).replace("/", "").replace("\\", "")
                blob.upload_from_string(json_str, content_type="text/json")
        elif file_extension == "json":
            blob.upload_from_string(json.dumps(obj))
        elif file_extension == "pkl":
            pickle_out = pickle.dumps(obj)
            blob.upload_from_string(pickle_out)
            logging.info("object is uploaded using pickle")
        elif file_extension in ["yaml", "yml"]:
            yaml_str = yaml.dump(obj)
            blob.upload_from_string(yaml_str, content_type="application/x-yaml")


def get_file_extension(filename: str) -> str:
    """Retrieve the file extension from the given filename."""
    return filename.split(".")[-1].lower()


def remove_from_gcs(BUCKET_PATH, folder):
    """Remove all files from GCS directory

    Args:
        gcs_path (:obj:`str`): full path including bucket name + blob name + filename

    Returns:
        (:obj:`data`)
    """
    gcs_path = BUCKET_PATH + folder
    file_path_split = gcs_path.split("/", 1)
    GS_BUCKET_NAME = file_path_split[0]

    GS_BLOB_NAME = file_path_split[-1]

    storage_client = storage.Client("tlb-data-dev")

    bucket = storage_client.get_bucket(GS_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GS_BLOB_NAME)
    # delete all files in directory including the subdirectory
    for blob in blobs:
        blob.delete()

    # GCS doesn't differentiate filenames from directories
    # this step will create the deleted folder
    blob = bucket.blob(folder)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    return
