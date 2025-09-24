# Use an official Python runtime as a parent image.
FROM python:3.11-slim

# Set Environment variables
ENV MINICONDA_VERSION=py311_24.3.0-0 \
    PATH=/opt/conda/bin:$PATH

ARG GCP_PROJECT
ENV PROJECT_ID=$GCP_PROJECT
RUN echo "RUN TIME PROJECT_ID =$PROJECT_ID"


# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary tools
RUN echo "Install necessary tools" && apt-get update && apt-get install -y git curl bzip2 gcc build-essential\
    && apt-get clean \
    && /bin/bash -c 'rm -rf /var/lib/apt/lists/*'

# Install Miniconda
RUN echo "Install Miniconda" && curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean --all -y

# Install the Google Artifact Registry keyring to handle authentication to the private package repository
# It is essential to authenticate to the Google Artifact Registry to install private packages.
#RUN pip install keyrings.google-artifactregistry-auth

# Installing private packages from Google Artifact Registry using pip worked with no GOOGLE_APPLICATION_CREDENTIALS explicitly set
# Example: RUN pip install tdata-utils --extra-index-url https://europe-west2-python.pkg.dev/{$PROJECT_ID}/tdata-utils/simple/


# Install private packages from Google Artifact Registry using Poetry
# It was able to authenticate but it takes a long time to install the package and never finishes. stuck at "Resolving dependencies..."
#RUN poetry source add --priority=explicit data_utils https://europe-west2-python.pkg.dev/{$PROJECT_ID}/tdata-utils/simple/
#RUN poetry add --source data_utils tdata-utils

# Challange to resolve:
# Setting the right project id for the private package repository while using poetry!

# Copy the entire directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Environment variable for Google Application Credentials
# DEV
# For tlb-data-staging-algo-cluster-1 the service account is:
# airflow-gcr@tlb-data-dev.iam.gserviceaccount.com

# Prod
# For tlb-data-prod-algo-cluster-prod the service account is:
# algo-cluster-airflow@tlb-data-prod.iam.gserviceaccount.com
ENV GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/service-account.json
