import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)
# MLflow client setup
client = mlflow.MlflowClient()

BQ_PROJECT_ID = "tlb-data-dev"

def load_conda_requirements(conda_path: Union[str, Path]) -> Tuple[List[str], List[str], str]:
    """Loads and parses conda.yaml, returning Conda dependencies, Pip dependencies, and Python version."""
    with open(conda_path, "r") as file:
        conda_env = yaml.safe_load(file)

    conda_requirements = []
    pip_requirements = []
    python_version = None

    # Parse dependencies
    for dep in conda_env.get("dependencies", []):
        if isinstance(dep, dict) and "pip" in dep:
            for pip_dep in dep["pip"]:
                if pip_dep.startswith("-r "):
                    req_file = pip_dep.split(" ", 1)[1]
                    req_file_path = Path(conda_path).parent / req_file
                    with open(req_file_path, "r") as req_file_content:
                        pip_requirements.extend(
                            [line.strip() for line in req_file_content if line.strip() and not line.startswith("#")]
                        )
                else:
                    pip_requirements.append(pip_dep)
        else:
            # Add non-pip dependencies to conda_requirements
            conda_requirements.append(dep)
            # Extract Python version if present
            if "python" in str(dep).lower():
                python_version = dep

    return conda_requirements, pip_requirements, python_version


def compare_environments(mlflow_logged_conda_env: Tuple[List[str], List[str], str],
                         source_code_conda_env: Tuple[List[str], List[str], str]) -> \
        Dict[str, List[str]]:
    """Compares two Conda environments, including dependencies and Python version, returning any mismatches."""
    mlflow_logged_conda_reqs, mlflow_logged_pip_reqs, mlflow_logged_python_version = mlflow_logged_conda_env
    source_code_conda_reqs, source_code_pip_reqs, source_code_python_version = source_code_conda_env
    report = {"mlflow_logged_conda_reqs": mlflow_logged_conda_reqs, "source_code_conda_reqs": source_code_conda_reqs,
              "mlflow_logged_pip_reqs": mlflow_logged_pip_reqs, "source_code_pip_reqs": source_code_pip_reqs,
              "mlflow_logged_python_version": mlflow_logged_python_version,
              "source_code_python_version": source_code_python_version}
    mismatches = {}

    # Compare Conda dependencies
    missing_in_conda_mlflow_logged_env = [dep for dep in source_code_conda_reqs if dep not in mlflow_logged_conda_reqs]
    missing_in_conda_source_code_env = [dep for dep in mlflow_logged_conda_reqs if dep not in source_code_conda_reqs]
    if missing_in_conda_mlflow_logged_env:
        mismatches["missing_in_conda_mlflow_logged_env"] = missing_in_conda_mlflow_logged_env
    if missing_in_conda_source_code_env:
        mismatches["missing_in_conda_source_code_env"] = missing_in_conda_source_code_env

    # Compare Pip dependencies
    missing_in_pip_mlflow_logged_env = [dep for dep in source_code_pip_reqs if dep not in mlflow_logged_pip_reqs]
    missing_in_pip_source_code_env = [dep for dep in mlflow_logged_pip_reqs if dep not in source_code_pip_reqs]
    if missing_in_pip_mlflow_logged_env:
        mismatches["missing_in_pip_mlflow_logged_env"] = missing_in_pip_mlflow_logged_env
    if missing_in_pip_source_code_env:
        mismatches["missing_in_pip_source_code_env"] = missing_in_pip_source_code_env

    # Compare Python versions
    if mlflow_logged_python_version != source_code_python_version:
        mismatches["python_version_mismatch"] = [f"model: {mlflow_logged_python_version}",
                                                 f"source: {source_code_python_version}"]

    print(f"    - Full ENV Deatils: {json.dumps(report, indent=2)}")

    return mismatches


def get_registered_models() -> List[str]:
    """Fetches all registered models from the MLflow Model Registry."""
    return [model.name for model in client.search_registered_models()]


def find_python_env_yaml_paths(local_path: str) -> Dict[str, Path]:
    """Finds paths for model's and source's conda.yaml files by scanning subdirectories."""
    model_python_env_path = Path(local_path) / "python_env.yaml"
    src_code_base = Path(local_path) / "code/projects"
    src_python_env_path = None
    model_dir = None

    if src_code_base.exists():
        # Recursively search for conda.yaml within subdirectories under projects
        for project_dir in src_code_base.iterdir():
            if project_dir.is_dir():
                for model_dir in project_dir.iterdir():
                    if model_dir.is_dir():
                        potential_python_env_path = model_dir / "python_env.yaml"
                        if potential_python_env_path.exists():
                            src_python_env_path = potential_python_env_path
                            break
                if src_python_env_path:
                    break

    if not src_python_env_path:
        print(f"Could not find a valid python_env.yaml file under {src_code_base}")
        raise FileNotFoundError("python_env.yaml file not found in source code directory.")

    return {"model_python_env_path": model_python_env_path, "src_python_env_path": src_python_env_path}


def validate_model_version(model_name: str, version: str):
    """Validates the model version by comparing conda.yaml files in model and src artifacts."""
    try:
        print(f"Validating {model_name} version {version}")
        artifact_uri = client.get_model_version_download_uri(model_name, version)
        print(f"    - Artifact URI: {artifact_uri}")

        # Create a temporary directory and download artifacts
        temp_dir = tempfile.mkdtemp()
        print(f"    - Downloading artifacts to: {temp_dir}")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=temp_dir
        )
        print(f"    - Artifacts downloaded to: {local_path}")
        paths = find_python_env_yaml_paths(local_path)
        model_python_env_path = paths["model_python_env_path"]
        src_python_env_path = paths["src_python_env_path"]

        model_conda_env = load_conda_requirements(model_conda_path)
        src_conda_env = load_conda_requirements(src_conda_path)
        conda_mismatches = compare_environments(model_conda_env, src_conda_env)
        if conda_mismatches:
            print(f"  - conda.yaml mismatches found:")
            for key, items in conda_mismatches.items():
                print(f"    - {key}: {items}")
        else:
            print("  - conda.yaml files match.")

        return conda_mismatches
    except Exception as e:
        print(f"  - Error validating model {model_name}, version {version}: {e}")
        return {"error": str(e)}


def main():
    models = client.list_registered_models()
    for model in models:
        model_name = model.name
        latest_versions = client.get_latest_versions(model_name)
        for version in latest_versions:
            env_mismatches = validate_model_version(model_name, version.version)
            if env_mismatches:
                write_mismatches_to_bigquery(env_mismatches, model_name, version.version, "data_playground.ml_models_env_mismatches")


def target_model_validator(model_name, version):
    env_mismatches = validate_model_version(model_name, version)
    if env_mismatches:
        write_mismatches_to_bigquery(env_mismatches, model_name, version, "data_playground.ml_models_env_mismatches")


def write_mismatches_to_csv(mismatches: Dict[str, List[str]], output_path: str):
    """Writes mismatches to a CSV file."""
    import csv

    with open(output_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Mismatch Type", "Dependency"])
        for key, items in mismatches.items():
            for item in items:
                writer.writerow([key, item])


def write_mismatches_to_GCS(mismatches: Dict[str, List[str]], bucket_name: str, output_path: str):
    """Writes mismatches to a CSV file in Google Cloud Storage."""
    import csv
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_path)

    with blob.open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Mismatch Type", "Dependency"])
        for key, items in mismatches.items():
            for item in items:
                writer.writerow([key, item])


# Write mismatches to BigQuery
# Write model name, version, validation timestamp, and mismatches to a BigQuery table
# Create table if it doesn't exist
# Partationed by validation_timestamp, clustered by model_name
def write_mismatches_to_bigquery(mismatches: Dict[str, List[str]], model_name: str, version: str, full_table_name: str):
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    from datetime import datetime

    client = bigquery.Client()
    table_id = f"{BQ_PROJECT_ID}.{full_table_name}"
    validation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Define schema
    schema = [
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("version", "STRING"),
        bigquery.SchemaField("mismatch_type", "STRING"),
        bigquery.SchemaField("dependency", "STRING"),
        bigquery.SchemaField("validation_timestamp", "TIMESTAMP")
    ]

    # Check if table exists, if not, create it with partitioning and clustering
    try:
        client.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except NotFound:
        # Define table with partitioning and clustering
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field="validation_timestamp")
        table.clustering_fields = ["model_name"]
        table = client.create_table(table)
        print(f"Created table {table_id} with partitioning on validation_timestamp and clustering on model_name.")

    # Prepare rows for insertion
    rows = []
    for key, items in mismatches.items():
        for item in items:
            rows.append({
                "model_name": model_name,
                "version": version,
                "mismatch_type": key,
                "dependency": item,
                "validation_timestamp": validation_timestamp
            })

    # Insert rows
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Inserted {len(rows)} rows into {table_id}")





if __name__ == "__main__":
    # Scan all
    # main()

    # Target model
    # ENV Matches
    target_model_validator("example_iris_classifier_v1", 5)
    # ENV Mismatches
    target_model_validator("vendor_image_processing_upscaler", 10)
