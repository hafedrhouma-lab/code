# data-item-tagging

In collaboration with DH Food Science team, the goal of this project is to enrich the existing taxonomy with item tags required for several use cases.



_last modified_: 27/12/2023



| Project         | Overview                                |
|-----------------|-----------------------------------------|
| Objective       | Tag Talabat Items using Open AI Prompts |   
| Key Outcomes    |                                         | 
| Status          | `IN PROGRESS`                           | 
| Deployment Date | Q4 2023                                 |

## Table of contents

* [Problem Statement](#problem-statement)
* [Project Scope](#project-scope)
* [Timeline and Work Breakdown Structure](#timeline-and-work-breakdown-structure)
* [Constraints](#constraints)
* [Installation](#installation)

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

##  Timeline and Work Breakdown Structure
### Milestones and deadlines
| Milestone                           | Deadline   | Status        | Status Date |  
|-------------------------------------|------------|---------------|-------------|
| Test on Food Items (Zomato Data)    | 31/08/2023 | `DONE`        | 16/08/2023  |
| Test on Grocery Items (Sample Data) | 31/08/2023 | `DONE`        | 16/08/2023  |
| Deploying model for Food Items      | 31/08/2023 | `IN PROGRESS` | 16/08/2023  |
| Deploying model for Grocecry Items  | 30/09/2023 | `NOT STARTED` | 16/08/2023  |


## Project structure:

An overview of current package structure.

```py
```


## Constraints:

* Running a prompt using openai (chatGPT) takes time (many seconds)
* Given the millions of items we have, it will be very challenging to run the prompt on all the items (weeks of code running)
* We must simplify the datasets, by : 
  * Excluding items with obvious tags (for example if title contains 'chicken', it's non vegetarian)
  * Eliminating items having the exact same title (though different item_id)

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

### NFV Items

#### 1. Prepare the chatGPT4 requests
```bash
python -m src.cli.nfv.prepare prepare --execution_date='2023-08-24'
```

Operated Tasks:
* Read the data from BigQuery
* create individuals requests by creating prompt/item information combinations

:information_source: The request file is saved locally on the machine

#### 2. Run the prompts (asynchronous calls)
```bash
python -m src.cli.nfv.run_prompt run_prompt --execution_date='2023-08-24'
```

:information_source: Uses the previously created file.

Operated Tasks:
* Save the results in json file locally


#### 3. Predict NFV Items
```bash
python -m src.cli.nfv.parse_response parse_response --execution_date='2023-08-24'
```

:information_source: Parse the previously generated json file
