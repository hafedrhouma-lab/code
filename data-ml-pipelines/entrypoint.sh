#!/bin/bash
# This script expects at least two arguments:
# 1. "project_name/model_name"
# 2. The command to execute followed by its arguments

# Set the project ID for gcloud
PROJECT_ID=${PROJECT_ID}
echo "Setting project ID to $PROJECT_ID"
if [[ "$PROJECT_ID" == "tlb-data-prod" ]]; then
  echo "Production environment detected"
else
  echo "Development environment detected"
fi

echo "Current Python version:"
python --version

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 project_name/model_name command [command_args]"
    exit 1
fi

IFS='/' read -r project_name model_name <<< "$1"
shift

# Navigate to the specific project and model directory
cd /usr/src/app/projects/${project_name}/${model_name}

# Ensure pip is updated in the base environment
echo "Updating pip in base environment..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to update pip in base environment."
    exit 1
fi

# enforce mlflow version to the server version
REQUIREMENTS_FILE="requirements.txt"
MLFLOW_VERSION="mlflow==2.16.0"

# Check if mlflow is present in the requirements.txt
if grep -q "^mlflow==" "$REQUIREMENTS_FILE"; then
    # Check if it is the desired mlflow version
    if grep -q "^$MLFLOW_VERSION$" "$REQUIREMENTS_FILE"; then
        echo "$MLFLOW_VERSION is already present in the requirements.txt"
    else
        # If mlflow is there but with a different version, update to the desired version
        sed "s/^mlflow==.*/$MLFLOW_VERSION/" "$REQUIREMENTS_FILE" > requirements.tmp && mv requirements.tmp "$REQUIREMENTS_FILE"
        echo "Updated mlflow to $MLFLOW_VERSION in the requirements.txt"
    fi
else
    # If mlflow is not there, add the desired version to the requirements.txt
    echo "$MLFLOW_VERSION" >> "$REQUIREMENTS_FILE"
    echo "Added $MLFLOW_VERSION to the requirements.txt"
fi

## Capture base pip packages before environment setup
#echo "Capturing existing pip packages..."
#pip freeze | grep -v 'file://' > base_packages.txt

# Path to the conda.yaml file
CONDA_YAML="conda.yaml"

# Create conda environment using conda.yaml
echo "Creating Conda environment using $CONDA_YAML..."
conda env create -f $CONDA_YAML

# Extract environment name from conda.yaml
ENV_NAME=$(grep 'name:' $CONDA_YAML | awk '{print $2}')

echo "Activating environment $ENV_NAME"
# To solve the issue of 'CondaError: Run 'conda init' before 'conda activate''
# Do the following:
conda init bash
source activate base
conda activate $ENV_NAME

echo "Using python at $(which python)"
python --version

# Conda and private package installation
# > It is not possible to install private packages using conda, so we need to install them using pip
# > adding extra index url to conda config failed
# > we need to install the private packages using pip with the extra index url in each package.


# read file own_private_packages.txt and loop in each line to install the packages with its own extra index url
# example file tdata-utils --extra-index-url https://europe-west2-python.pkg.dev/PROJECT_ID/tdata-utils/simple/
# ignore if file not found
if [ -f own_private_packages.txt ]; then
    while IFS= read -r line || [ -n "$line" ]
    do
        # Skip empty lines
        [ -z "$line" ] && continue

        # Replace the PROJECT_ID placeholder with the runtime project ID
        line_with_project_id=$(echo "$line" | sed "s/PROJECT_ID/${PROJECT_ID}/g")
        echo "Installing package $line_with_project_id"
        pip install $line_with_project_id
    done < own_private_packages.txt
fi

echo "conda info.."
conda info

# Set PYTHONPATH to the specified directory relative to the current directory
export PYTHONPATH=../../..

# Execute the command passed as remaining arguments in the gke pod
"$@"
