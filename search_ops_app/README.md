# Gorcery Search Ops

_last modified_: 30/06/2022


| Project      | Overview                                                                       |
|--------------|--------------------------------------------------------------------------------|
| Objective    | To help improve grocery search performances                                    |   
| Key Outcomes | The app allows to generate automatically groups of Best/Worst performing query | 
| Status       | `IN PROGRESS`                                                                  | 
| MEP Date     | Q2 2022                                                                        |

## Table of contents

* [Problem Statement](#problem-statement)
* [Project Scope](#project-scope)
* [Timeline and Work Breakdown Structure](#timeline-and-work-breakdown-structure)
* [App constraints](#app-constraints)
* [App next steps](#app-next-steps)
* [Run the app](#run-the-app)
* [Releases](#releases)

## Problem Statement:

[(:pushpin: Read the doc)](https://jubilant-tribble-2e7a6828.pages.github.io)


Currently, searches grocery search performances are not optimal. This might be due to:
* Assortment issues
* Missing vendors
* Inexact meta data 
* Out of stocks items


## Project Scope:
* Must have:
    * An app allowing to display CTR, CVR, Search results, clicks for every searches in grocery, at area name granular level, for every country
* Nice to have:
    * Clustering searches and display performances by groups 
    * Make inference on search query to analyse if search are made on the right search tab
* Not in current scope:
    * Predict what should be the right product to show for a given search

##  Timeline and Work Breakdown Structure
### Milestones and deadlines
| Milestone                                | Deadline   | Status    | Status Date |  
|------------------------------------------|------------|-----------|-------------|
| Understand searches in grocery           | 10/04/2022 | `DONE`    | 23/04/2022  |
| Make proposal of app content             | 10/04/2022 | `DONE`    | 23/04/2022  |
| Validate data availaibility              | 17/04/2022 | `DONE`    | 23/04/2022  |
| App Prototype                            | 22/04/2022 | `DONE`    | 23/04/2022  |
| Release 1.0                              | 02/05/2022 | `DONE`    | 23/04/2022  |
| App deployment                           | 03/05/2022 | `DONE`    | 06/05/2022  |
| Release 2.0: Homepage queries clustering | 12/05/2022 | `DONE`    | 30/06/2022  |
| Release 3.0: In Vendor Search clustering | 01/07/2022 | `DONE`    | 30/06/2022  |
| Monitoring Loop                          | 30/12/2022 | `STARTED` ||
## Project structure:

An overview of current package structure.

```py
├── Dockerfile
├── Makefile
├── README.md
├── app.py
├── cloudbuild.yaml
├── docker-compose.yml
├── mkdocs.yml
├── requirements.txt
├── run.sh
├── src
│   ├── __init__.py
│   ├── app #App execution
│   │   ├── __init__.py
│   │   └── multipage.py
│   ├── conftest.py
│   ├── css
│   │   └── style.css
│   ├── data #Data module to handle all the data fetch and transformations
│   │   ├── __init__.py
│   │   ├── datasets # Construction of dataset: fetch
│   │   │   ├── __init__.py
│   │   │   ├── fetch.py
│   │   │   └── queries
│   │   │       └── aggregates
│   │   │           ├── agg_country_search_performances.sql.j2
│   │   │           └── agg_session_level_performances.sql.j2
│   │   └── processors # Preprocessing of the dataframes
│   │       ├── __init__.py
│   │       ├── country_metric.py
│   │       └── search_query.py
│   ├── model # Model module to train, test, evaluate 
│   │   ├── clustering
│   │   │   ├── __init__.py
│   │   │   └── clustering_quantile.py
│   │   └── interfaces.py
│   ├── pages # Contains all the streamlit components. e.g View
│   │   ├── __init__.py
│   │   ├── clustering
│   │   │   ├── __init__.py
│   │   │   ├── clusters.py
│   │   │   └── country_metric.py
│   │   └── resources # resources for app working
│   │       ├── country_city_area_mapping.csv
│   │       └── users_specifications.yaml
│   └── utils # Helper functions
│       ├── __init__.py
│       ├── exceptions.py
│       ├── helper.py
│       └── log.py
└── tests # UnitTest

```


## App constraints:
* We are currently limited to analyse search from homepage and from grocery homepage. 
* So far, in vendor search analyses are not possible as data are not available and must be obtained by ETL project on raw datas

## App next steps:
* Visualise similars searches within session with word clouds

## Run the app

| Language | Version |
|----------|---------|
| Python   | 3.8     |   


1. Using virtual env
```bash
> make venv
> export GOOGLE_APPLICATION_CREDENTIALS="path_to_key"

> make run 
```

2. Using Docker
```bash
> docker build -t search_ops_app:latest . # build
> docker run -d -p 8501:8501 search_ops_app:latest # run
> docker stop [CONTAINER_ID] # stop
```

Using docker will require as well to export `GOOGLE_APPLICATION_CREDENTIALS` in docker container at run step


## Releases

:arrow_up: 1.0 - 02/05/2022 - Query performances based on user filters

:arrow_up: 2.0 - 12/05/2022 - Generate automatically clusters of Best/Worst performing queries for **homepage search**

:arrow_up: 3.0 - 30/06/2022 - Generate automatically clusters of Best/Worst performing queries for **in Vendor search**

