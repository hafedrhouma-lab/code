from google.cloud import storage
import os
import logging

class GCSOperations:

    def __init__(self, local_folder_path, bucket_name, gcs_folder_path):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.local_folder_path = local_folder_path
        self.bucket_name = bucket_name
        self.gcs_folder_path = gcs_folder_path
        self.client = storage.Client()

    def sync_local_to_gcs(self):
        """
        Sync a local folder to a Google Cloud Storage folder.
        """

        bucket = self.client.get_bucket(self.bucket_name)

        # Get the list of existing blobs in the GCS folder
        existing_blobs = list(bucket.list_blobs(prefix=self.gcs_folder_path))

        # Dictionary to keep track of files that exist locally
        local_files = {}

        # Upload new and modified files
        for root, _, files in os.walk(self.local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, self.local_folder_path)
                gcs_blob_path = os.path.join(self.gcs_folder_path, relative_path).replace("\\", "/")

                local_files[gcs_blob_path] = True

                blob = bucket.blob(gcs_blob_path)
                blob.upload_from_filename(local_file_path)
                logging.info(f'Uploaded {local_file_path} to {gcs_blob_path}')

        # Clean up deleted files
        self.clean_deleted_files(existing_blobs, local_files)

    def clean_deleted_files(self, existing_blobs, local_files):
        """
        Delete files in GCS that no longer exist locally.
        """
        logging.info('Cleaning up deleted files from GCS.')
        for blob in existing_blobs:
            if blob.name not in local_files:
                blob.delete()
                logging.info(f'Deleted {blob.name} from GCS')