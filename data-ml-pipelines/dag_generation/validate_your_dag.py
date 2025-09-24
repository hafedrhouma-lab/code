import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date format
)


def validate_my_DAG_config(project_name, model_name):
    from dag_generator import DagGenerator
    schedule_yaml_files = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    projects_dir = os.path.join(script_dir, "..", "projects")
    out_put_dir = os.path.join(script_dir, "..", "tmp")
    project_path = os.path.join(projects_dir, project_name)
    model_path = os.path.join(project_path, model_name)
    yaml_file = os.path.join(model_path, "schedule.yaml")
    if os.path.exists(yaml_file):
        schedule_yaml_file = {"project_name": project_name, "model_name": model_name, "yaml_file": yaml_file}
        dag_gen = DagGenerator()
        dag_gen.auto_generated_dags_dir = out_put_dir
        if not dag_gen.generate_dag(schedule_yaml_file):
            print(f"Failed to generate DAG for {project_name} - {model_name}.")
        else:
            print(
                f"DAG for {project_name} - {model_name} generated successfully, please check the output directory {out_put_dir}")
    else:
        print(f"schedule.yaml file not found in {model_path}")


def install_package(package):
    print("Installing package: ", package)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def try_import(package, import_name):
    try:
        if import_name:
            globals()[import_name] = __import__(import_name)
    # except ModuleNotFoundError or ImportError
    except Exception as e:
        install_package(package)
        if import_name:
            globals()[import_name] = __import__(import_name)
        else:
            globals()[package] = __import__(package)
        print(f"Successfully installed and imported {package}")


def check_dependencies():
    packages = [
        ("pyyaml", "yaml"),
        ("google-cloud-storage", "google.cloud.storage"),
        ("cookiecutter", "cookiecutter"),
    ]
    for package, import_name in packages:
        try_import(package, import_name)


if __name__ == "__main__":
    # This function will help you valide you schedule.yaml file
    # It will generate a DAG based on the configuration in the schedule.yaml file
    # It will store the DAG in the tmp directory if the generation is successful

    parser = argparse.ArgumentParser(description="Process User project,model parameters.")
    parser.add_argument("project_name", type=str, help="The Project Name string parameter")
    parser.add_argument("model_name", type=str, help="The Model Name string parameter")
    args = parser.parse_args()
    if not args.project_name or not args.model_name:
        print("Please provide the project name and model name")
        print(
            "Project name and model name are required and has to be excatly the same as the directory name in the projects directory")
        print("Example: python validate_your_dag.py example iris_classifier_v0")
        exit(1)
    check_dependencies()
    validate_my_DAG_config(args.project_name, args.model_name)

    # check the tmp/ directory for the generated DAG inside this project directory.
