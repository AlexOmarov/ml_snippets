FROM nvidia/cuda:11.7.1-base-ubuntu20.04

WORKDIR /ml_snippets
ADD . /ml_snippets

# Make required directories
RUN mkdir -p /ml_snippets/data/logs
RUN mkdir -p /ml_snippets/data/models
RUN mkdir -p /ml_snippets/logs

# Since wget is missing
RUN apt-get update && apt-get install -y wget

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install gcc as it is missing in our base layer
RUN apt-get update && apt-get -y install gcc

#  Remove deps from environment, download it from setuptools
RUN conda config --set unsatisfiable_hints false
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ml_snippets", "/bin/bash", "-c"]

EXPOSE 5000

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "ml_snippets", "python", "src/app/app.py"]