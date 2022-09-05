FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

WORKDIR /od_docker
ADD . /od_docker

# Since wget is missing
RUN apt-get update && apt-get install -y wget

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install gcc as it is missing in our base layer
RUN apt-get update && apt-get -y install gcc

# Here all of the sources AND conda are already present
# We need to create new conda env, activate it, launch setup.py via anaconda pip
# Then we need to run this env
# So we need to get env.yml for anaconda initialization, but without any packages (just name, channels, setuptools, etc)
# Then, install all the packages via setup.py pip / python

RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN python -c "import flask"

EXPOSE 5000

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "src/my_app.py"]