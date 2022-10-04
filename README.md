# Ml snippets service

Service for testing ML samples, mathematical conceptions, writing PoC solutions.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=AlexOmarov_ml_snippets_service&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=AlexOmarov_ml_snippets_service)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=AlexOmarov_ml_snippets_service&metric=coverage)](https://sonarcloud.io/summary/new_code?id=AlexOmarov_ml_snippets_service)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=AlexOmarov_ml_snippets_service&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=AlexOmarov_ml_snippets_service)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=AlexOmarov_ml_snippets_service&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=AlexOmarov_ml_snippets_service)
[![Python version](https://img.shields.io/static/v1?label=Python&message=3.9&color=blue)](https://sonarcloud.io/summary/new_code?id=AlexOmarov_ml_snippets_service)

## Table of Contents

- [Introduction](#introduction)
- [Documentation](#documentation)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)

## Introduction

Ml snippets service is a service which serves as a container of PoC machine learning solutions.  
It also includes:

* various snippets;
* storage for ml algorithms results and datasets.
  All functionality is available via http endpoints (see [Documentation](#documentation))

## Documentation

All the service's documentation can be found in [docs](docs) folder
There you can find:

- Data model
- Class diagram
- Sequence diagram for all the business flows which are performed by service
- Description of all service integrations
- Descriptions of service API's, incoming and outcoming esb events
- Some additional diagrams for better understanding of underlying processes

## Features

* Audio analysis, feature extraction
* Mnist digits image classification

## Requirements

The application can be run locally or in a docker container,
the requirements for each setup are listed below.

### Local

* [Miniconda >= 3](https://conda.io/en/latest/miniconda.html)
* Python 3.9

### Docker

* [Docker](https://www.docker.com/get-docker)

## Quick Start

Application will run by default on port `5000`

Configure the port by changing `SERVER_PORT` in __resource/config.py__

### Run Local

First, create new conda env from [unix_environment.yaml](unix_environment.yaml)
Depending on which OS you a working, environment files may differ

* unix_environment.yaml
* win_environment.yaml
*

```bash
# for unix
conda env update --file unix_environment.yaml
```

You can run application either via PyCharm launch configuration (preferred way) or manually:

```bash
$ conda activate ml_snippets_service
# Here insert into PYTHONPATH env paths to src.main.app folder
$ python -m src.main.app.app.py
```

When launching from PyCharm make sure you choose newly created conda interpreter and mark
`Add contents / sources roots to PYTHONPATH`

### Run Docker

Use `docker-compose-local.yml` to build the image and create a container.

### Run code quality assurance tasks

If you want to get total coverage with local changes, then you should run following task:

```bash
python -m pytest --cov -s -v src/test/app/tests.py --junitxml coverage.xml
```

Then, xml test report with coverage will be generated on local machine in root folder.

To get code quality, install [SonarLint](https://plugins.jetbrains.com/plugin/7973-sonarlint)
Then right click on project root folder and choose SonarLint -> Analyse

## API

### Web endpoints

All the service's web endpoints specification can be found on [docs](docs/api)

### Esb events

All the service's events specification can be found on [docs](docs/events)