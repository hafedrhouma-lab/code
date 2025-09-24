import os
import logging
import pandas as pd
import numpy as np
import json
import time
from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth import default


class BigQuery:
    def __init__(self, json_path=None, project=None, use_query_cache=True):
        """Specify the credentials you want to use.

        Args:
            json_path (:obj:`str`): path to the json file containing the credentials
            project (:obj:`str`): project id
            use_query_cache (:obj:`bool`): use query cache
        """

        # From json file
        if json_path:
            self.credentials = service_account.Credentials.from_service_account_file(
                json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        # From environment variable
        else:
            try:
                self.credentials = (
                    service_account.Credentials.from_service_account_info(
                        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
                logging.info("GOOGLE_APPLICATION_CREDENTIALS is used - account info")

            except Exception as e:
                try:
                    self.credentials = (
                        service_account.Credentials.from_service_account_file(
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                            scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        )
                    )
                    logging.warning(e)
                    logging.info(
                        "GOOGLE_APPLICATION_CREDENTIALS is used - account file"
                    )
                # Fallback to User authentication
                except Exception as e:
                    try:
                        # From user's credentials (Application Default Credentials)
                        self.credentials, _ = default()
                        logging.info("User account credentials are used.")
                    except Exception as e:
                        self.credentials = None
                        logging.warning(e)
                        logging.warning(
                            "GOOGLE_APPLICATION_CREDENTIALS is not set and User account credentials are not available! credentials are set to None"
                        )

        try:
            if project is None:
                logging.info("Project got from credentials")
                self.project = self.credentials.project_id

            else:
                logging.info("Project got from user input")
                self.project = project

        except:  # to handle None credentials
            logging.info(
                "Project was set to dev (default), since no credentials was passed"
            )
            self.project = "tlb-datateam-analysis-6343"

        logging.info("Project:" + self.project)

        self.client = bigquery.Client(
            credentials=self.credentials, project=self.project
        )
        self.job_config = bigquery.job.QueryJobConfig(
            allow_large_results=True, use_query_cache=use_query_cache
        )

    def read(
        self,
        query,
        create_bqstorage_client=True,
        optimize_download=True,
        progress_bar=True,
    ):
        """Reads a table from  big query database in tlb-data-prod / tlb-data-dev project based on the query and project_id passed

        Args:
            query (:obj:`str`): query to be executed
            create_bqstorage_client (:obj:`bool`): use the BQ storage API
            optimize_download (:obj:`bool`): optimize the download
            progress_bar (:obj:`bool`): show a progress bar

        Returns:
            :obj:`dataframe`: dataframe containing the query
        """
        st = time.time()
        try:
            logging.info("Google API is used")
            df = self.client.query(
                query, project=self.project, job_config=self.job_config
            ).result()
            if optimize_download:
                if progress_bar:
                    df = df.to_arrow(
                        create_bqstorage_client=create_bqstorage_client,
                        progress_bar_type="tqdm",
                    ).to_pandas()
                else:
                    df = df.to_arrow(
                        create_bqstorage_client=create_bqstorage_client
                    ).to_pandas()
            else:
                if progress_bar:
                    df = df.to_dataframe(
                        create_bqstorage_client=create_bqstorage_client,
                        progress_bar_type="tqdm",
                    )
                else:
                    df = df.to_dataframe(
                        create_bqstorage_client=create_bqstorage_client
                    )

        except:
            logging.warning("(Google API has failed")
            logging.info("Pandas API is used")
            df = pd.read_gbq(
                query,
                project_id=self.project,
                dialect="standard",
                use_bqstorage_api=create_bqstorage_client,
                credentials=self.credentials,
            )

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        logging.info("Execution time: " + str(np.round(elapsed_time, 2)) + " seconds")

        return df

    def write(self, df, table_name, if_exist_action, schema=[]):
        """writes a dataframe to a big query table under the tlb-data-prod / tlb-data-dev project based on the project_id passed

        Args:
            df (:obj:`dataframe`): dataframe to be published
            table_name (:obj:`str`): name of the table to be created
            if_exist_action (:obj:`str`): action to be taken if the table already exists
            schema (:obj:`list`): schema of the table to be created
                - Example:
                    >>> schema = [
                        bigquery.SchemaField("datetime", bigquery.enums.SqlTypeNames.TIMESTAMP),
                        bigquery.SchemaField("title", bigquery.enums.SqlTypeNames.STRING)
                        ]
                - Possible values for columns type:
                    * BIGDECIMAL = 'BIGNUMERIC'
                    * BIGNUMERIC = 'BIGNUMERIC'
                    * BOOL = 'BOOLEAN'
                    * BOOLEAN = 'BOOLEAN'
                    * BYTES = 'BYTES'
                    * DATE = 'DATE'
                    * DATETIME = 'DATETIME'
                    * DECIMAL = 'NUMERIC'
                    * FLOAT = 'FLOAT'
                    * FLOAT64 = 'FLOAT'
                    * GEOGRAPHY = 'GEOGRAPHY'
                    * INT64 = 'INTEGER'
                    * INTEGER = 'INTEGER'
                    * INTERVAL = 'INTERVAL'
                    * NUMERIC = 'NUMERIC'
                    * RECORD = 'RECORD'
                    * STRING = 'STRING'
                    * STRUCT = 'RECORD'
                    * TIME = 'TIME'
                    * TIMESTAMP = 'TIMESTAMP'
        """
        gAPI_disposition_translation = {
            "fail": "WRITE_EMPTY",
            "replace": "WRITE_TRUNCATE",
            "append": "WRITE_APPEND",
        }
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=gAPI_disposition_translation[if_exist_action],
        )

        job = self.client.load_table_from_dataframe(
            df, table_name, job_config=job_config
        )  # Make an API request.
        job.result()  # Wait for the job to complete.

        table = self.client.get_table(table_name)  # Make an API request.
        logging.info("Google API is used")
        logging.info(
            "Uploaded "
            + str(table.num_rows)
            + " rows and "
            + str(len(table.schema))
            + " columns to "
            + table_name
        )

        return

    def send_dml(self, statement):
        """Send a DML statement to be executed

        Example:
            >>> DELETE FROM data_playground.ese_chain_embeddings WHERE TRUE
        """
        query_job = self.client.query(statement)  # API request
        query_job.result()  # Waits for statement to finish
