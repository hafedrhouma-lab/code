import os
import sys
import json
import yaml
import logging
import re
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_name)
        logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        logging.error(
            f"Failed to upload {source_file_name} to {destination_blob_name}: {e}"
        )
        sys.exit(1)


def set_defaults(dag_config):
    """Sets default values for missing but required configurations and handles sql_sensor settings."""
    if "node_pool" not in dag_config:
        dag_config["node_pool"] = "algo_generic_node_pool"
    if "image" not in dag_config:
        dag_config["image"] = (
            f"gcr.io/{os.getenv('PROJECT_ID', 'tlb-data-dev')}/algo-master-image:latest"
        )

    if "sql_sensors" in dag_config and type(dag_config["sql_sensors"]) == list:
        sensors = []
        for sensor_data in dag_config["sql_sensors"]:
            for index, sensor in sensor_data.items():

                required_sensor_keys = [
                    "sensor_priority",
                    "table_name",
                    "dataset_name",
                    "sensor_properties",
                ]
                missing_keys = [key for key in required_sensor_keys if key not in sensor]
                if missing_keys:
                    logging.error(f"Missing required sql_sensor keys: {missing_keys}")

                props = sensor.get("sensor_properties", {})
                if "event_date_column" not in props:
                    logging.error("Missing 'event_date_column' in sql_sensor properties")
                props["event_date"] = validate_and_process_date(
                    props.get("event_date", "current_date()-1")
                )
                props["lower_bound"] = props.get("lower_bound", 10)

                sensor["sensor_properties"] = props
                sensors.append(sensor)
        dag_config["sql_sensors"] = sensors


def validate_and_process_date(date_str):
    """Validates the event_date format and keeps it if it's a valid SQL expression, or defaults it."""
    # Regular expression to match "current_date()-X" where X is a number of days
    pattern = r"^current_date\(\)(-\d+)?$"
    if re.match(pattern, date_str):
        return date_str
    else:
        logging.error(
            f"Invalid date format in event_date: {date_str}, defaulting to current_date()-1"
        )
        return "current_date()-1"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..", "projects")
    manifest = {}

    for project_name in os.listdir(root_dir):
        project_path = os.path.join(root_dir, project_name)
        if os.path.isdir(project_path):
            for model_name in os.listdir(project_path):
                model_path = os.path.join(project_path, model_name)
                if os.path.isdir(model_path):
                    yaml_file = os.path.join(model_path, "schedule.yaml")
                    if os.path.exists(yaml_file):
                        local_manifest = {}
                        with open(yaml_file, "r") as file:
                            try:
                                yaml_content = yaml.safe_load(file)
                                for dag_key, dag_config in yaml_content.items():
                                    # dag_prefix = f"{project_name}_{model_name}_{dag_key}"
                                    dag_prefix = dag_key
                                    if dag_prefix in local_manifest:
                                        logging.error(
                                            f"Duplicate DAG key {dag_prefix} found. Overwriting with latest configuration."
                                        )
                                    set_defaults(dag_config)
                                    local_manifest[dag_prefix] = {
                                        "project_name": project_name,
                                        "model_name": model_name,
                                        **dag_config,
                                    }
                                    logging.info(
                                        f"Project {project_name}, Model {model_name}, Dag {dag_key} configuration successful."
                                    )
                            except yaml.YAMLError as exc:
                                logging.error(
                                    f"Error parsing YAML file {yaml_file}: {exc}"
                                )
                        manifest.update(local_manifest)
                    else:
                        logging.info(f"'schedule.yaml' not found in {model_path}.")

    manifest_json_path = "manifest.json"
    with open(manifest_json_path, "w") as json_file:
        json.dump(manifest, json_file, indent=4)

    project_id = os.getenv("PROJECT_ID", "tlb-data-dev")
    bucket_name = f"{project_id}-data-algorithms-content-optimization"
    upload_to_gcs(
        bucket_name=bucket_name,
        source_file_name=manifest_json_path,
        destination_blob_name="mlflow_dags/ml_dags_schedule_manifest.json",
    )


if __name__ == "__main__":
    main()
