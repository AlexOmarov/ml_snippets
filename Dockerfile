FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Since wget is missing
RUN apt-get update && apt-get install -y wget

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1

# Install gcc as it is missing in our base layer
RUN apt-get update && apt-get -y install gcc

#  Create conda env
RUN conda config --set unsatisfiable_hints false

EXPOSE 5000

# Service specific commands
ENV PYTHONPATH /ml_snippets_service:$PYTHONPATH

WORKDIR /ml_snippets_service

ADD ./src/main /ml_snippets_service/src/main
ADD unix_environment.yaml /ml_snippets_service/unix_environment.yaml
ADD pyproject.toml /ml_snippets_service/pyproject.toml

# Make required directories
RUN mkdir -p /ml_snippets_service/data/logs
RUN mkdir -p /ml_snippets_service/data/models
RUN mkdir -p /ml_snippets_service/logs

RUN conda env create -f unix_environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ml_snippets_service", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "ml_snippets_service", "python", "src/main/app/app.py"]