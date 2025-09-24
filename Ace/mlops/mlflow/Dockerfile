FROM continuumio/miniconda3:23.10.0-1

# Install cmd tool
RUN apt-get update --quiet && \
    apt-get install --quiet --yes software-properties-common openssh-client git bat vim

# Install python packages
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install jupyterlab notebook -y

WORKDIR /mlops
