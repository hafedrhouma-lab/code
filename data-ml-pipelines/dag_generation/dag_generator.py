import json
import logging
import os

import yaml
from cookiecutter.main import cookiecutter

from gcs_operations import GCSOperations


class DagGenerator:
    """
    A class responsible for generating Airflow DAGs from templates and YAML configurations.

    Attributes:
        script_dir (str): Directory of the script executing this class.
        projects_dir (str): Directory containing project configurations.
        auto_generated_dags_dir (str): Directory where generated DAGs are stored.
        bucket_name (str): Google Cloud Storage bucket name for storing DAGs.
        schedule_yaml_files (list): List of dictionaries containing project, model names, and their corresponding schedule.yaml file paths.
        gcs_operations (GCSOperations): Instance for interacting with Google Cloud Storage.
    """

    def __init__(self):
        """
        Initializes the DagGenerator with default or specified template name.

        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.projects_dir = os.path.join(self.script_dir, "..", "projects")
        self.auto_generated_dags_dir = os.path.join(self.script_dir, "auto_generated_dags")
        self.bucket_name = self._set_bucket_name()
        self.schedule_yaml_files = []
        self.gcs_operations = None

    def initialize_gcs_operations(self):
        """Initializes the GCSOperations instance for interacting with Google Cloud Storage."""
        return GCSOperations(
            self.auto_generated_dags_dir,
            self.bucket_name,
            "dags/mlflow_dags/auto_generated"
        )

    def _set_bucket_name(self):
        """
        Sets the bucket name based on the environment variable 'PROJECT_ID'.
        Uses a production bucket if 'PROJECT_ID' is 'tlb-data-prod', otherwise uses a staging bucket.
        """
        if os.getenv('PROJECT_ID') == 'tlb-data-prod':
            logging.warning("Setting bucket name to production bucket.")
            return "europe-west2-tlb-data-prod--b68151e3-bucket"
        else:
            logging.info("Setting bucket name to staging bucket.")
            return "europe-west2-tlb-data-stagi-1e59d66d-bucket"

    def generate_dag(self, file_config_objct: dict):
        """
        Generates a DAG by reading a YAML configuration file and using a cookiecutter template.

        Args:
            config_objct (duct): YAML configuration file.

        Returns:
            bool: True if the DAG was successfully generated, False otherwise.
        """
        try:
            yaml_input_path = file_config_objct['yaml_file']

            with open(yaml_input_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            for dag_key, dag_config in yaml_content.items():
                enable_dag = dag_config.get('enable_dag', False)
                if not enable_dag:
                    logging.warning(f"DAG {dag_key} is disabled. Skipping generation.")
                    logging.warning(f"To Enable this DAG {dag_key}, update the schedule.yaml file and set enable_dag to True")
                    continue
                dag_config['dag_id'] = dag_key
                dag_config["project_name"] = file_config_objct["project_name"]
                dag_config["model_name"] = file_config_objct["model_name"]
                print(f"Generating DAG for {dag_key}")
                print(json.dumps(dag_config, indent=4))
                print("=========================================")
                # if template name in dag_config, use it, otherwise use the default template
                template_name = dag_config.get('template_name', 'gke_pod_with_sql_sensors')
                self.template_path = os.path.join(self.script_dir, 'dags_templates', template_name)
                tasks_group = dag_config.get('tasks_group')
                dag_config['tasks_group'] = {"DAG_TASKS_GROUP": tasks_group}
                cookiecutter(self.template_path, output_dir=self.auto_generated_dags_dir, extra_context=dag_config,
                             no_input=True)
            return True
        except Exception as e:
            logging.error(f"Failed to generate DAG: {e}")
            return False

    def find_schedule_yaml_files(self):
        """
        Scans the projects directory for 'schedule.yaml' files and stores their paths.

        This method updates the `schedule_yaml_files` attribute with found files.
        """
        for project_name in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, project_name)
            if os.path.isdir(project_path):
                for model_name in os.listdir(project_path):
                    model_path = os.path.join(project_path, model_name)
                    yaml_file = os.path.join(model_path, "schedule.yaml")
                    if os.path.exists(yaml_file):
                        self.schedule_yaml_files.append(
                            {"project_name": project_name, "model_name": model_name, "yaml_file": yaml_file})

    def start(self):
        """
        Initiates the DAG generation process.

        This method scans for 'schedule.yaml' files, generates DAGs based on them, and syncs the generated DAGs to GCS.
        """
        self.find_schedule_yaml_files()
        for schedule_yaml_file in self.schedule_yaml_files:
            if not self.generate_dag(schedule_yaml_file):
                logging.error(
                    f"Failed to generate DAG for {schedule_yaml_file['project_name']} - {schedule_yaml_file['model_name']}.")
        self.gcs_operations = self.initialize_gcs_operations()
        self.gcs_operations.sync_local_to_gcs()


if __name__ == "__main__":
    dag_generator = DagGenerator()
    dag_generator.start()