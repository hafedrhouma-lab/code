# data-item-tagging

This project contains code to train a [roberta model](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) and predict item tags using the obtained model.


_last modified_: 27/12/2023


| Project         | Overview                                                    |
|-----------------|-------------------------------------------------------------|
| Objective       | Create models for Talabat item Taggings (food and non food) |   
| Key Outcomes    |                                                             | 
| Status          | `IN PROGRESS`                                               | 
| Deployment Date | Q4 2023                                                     |

## Table of contents

* [Problem Statement](#problem-statement)
* [Project Scope](#project-scope)
* [Constraints](#constraints)
* [Installation](#installation)
* [Usage](#usage)
* [Releases](#releases)

## Problem Statement:

* Looking for a certain item can be time consuming and discouraging given the thousands of choice we offer at Talabat.
* Having appropriates tags for items will simplify the items research 

## Project Scope:
* Must have:
    * A model that attribute relevant tags to an item among a set of possible values 
* Nice to have:
    * `TBD`
* Not in current scope:
    * `TBD`


## Constraints:


## Installation
```bash
pip install .
```

## Usage

<img src="https://img.shields.io/badge/python->3.8-green" />

* As we rely on google big query to acess the row datasets, check that google credentials are defined in the environment variables.
* We also need to define an openAI key as we use their [library](https://github.com/openai/openai-python)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path_to_key"
export OPENAI_API_KEY=your_key_value
```


### Food Items

:information_source: The labelled dataset can be obtained by a LLM output, or by human annotation

#### 1. Train the model
```bash
python -m src.cli.food.train train --execution_date='2023-08-24'
```

:information_source: Output model is saved locally. For deployment, GCP buckets should be the destination.


#### 2. Predict food Items
```bash
python -m src.cli.food.predict predict  --text 'pizza bolognese'
```

==> Returns `non_vegetarian`


### NFV Items

:information_source: The labelled dataset can be obtained by a LLM output, or by human annotation

#### 1. Prepare the data (one hot encoded labels etc)
```bash
python -m src.cli.nfv.prepare prepare --execution_date='2023-08-24'
```
:information_source: The training dataset is saved locally. For deployment purposes, GCP buckets should be used.

#### 2. Train the model
```bash
python -m src.cli.nfv.train train --execution_date='2023-08-24'
```
:information_source: The obtained model is saved locally. For deployment purposes, GCP buckets should be used.

#### 3. Predict NFV Items
```bash
python -m src.cli.nfv.predict predict --text 'pasta box with vegetables'
```

==> Returns 

`vegetarian
vegan
ready_to_eat
plant_based
`

## Releases (TODO)

:arrow_up: 1.0 - --/--/2023 - TBD
