# tLabel using LLM

## Contents
1. [Introduction](#introduction)
2. [Workflow](#workflow)
   1. [System prompt](#1-system-prompt)
   2. [Chain menu](#2-chain-menu)
   3. [DAGs](#3-dags)
   4. [Saving results](#4-saving-results)
   5. [MainDB](#5-maindb)
3. [Notes](#notes)
   1. [Running the code](#running-the-code)
   2. [LLM token](#llm-token)
   3. [Running the code locally](#running-the-code-locally)


<BR> The goal is to build a flexible and scalable chain tagging system leveraging the capabilities of LLMs to enrich restaurants with tags in different categories to be used in different real-estates in the app.

<BR> The use of LLMs enables dynamic tagging aligned with evolving business requirements without the need for manual labeling or retraining a whole new algorithm for the initial tagging or with the addition of further tags.

## Workflow
The following diagram illustrates the workflow of the tagging system

![hl_workflow.jpg](images%2Fhl_workflow.jpg)


### 1. System prompt
The system prompt is set per a combination of the (category & country).
* **The category** defines the system prompting and the LLM scope. There is a prompt per category under the **prompts directory**
* **The country** defines the tags that are relevant to this market. The tags are set in the **tags_mapping** **yaml** file under the **tags directory**

At runtime, both the prompt and the relevant tags are combined to generate the final prompt.

### 2. Chain menu
The chain menu represent the user prompt. 
<br>At runtime, The chain menus are fetched in batches from BQ using the **sql/chains_menu.sql**, processed in the following format, then fed to the LLM to return the suitable tags.
```bash
chain_id: [
   <category> -- <item_name> -- <description> -- <order_percentage>
   ...
]
```
* At most, 5 chains are processed in one prompt to avoid long prompts and to ensure the LLM can handle the input.
* The output format is defined per category using **pydantic** in **output_formats.py**. Ex: GeographicalUnitResponse
```python
class GeographicalUnitResponse(BaseModel):
    chain_id: str
    tag_1: str = Field(..., description="Geographical cuisine tag")
    explanation_tag_1: str = Field(..., description="Reason for cuisine choice (max 100 chars)")
```
### 3. DAGs
Two DAGs are set per category and country pair. 
1. The first DAG is for initial load, triggered manually to overwrite the existing tags for all chains of the market under this category.
   * Uses sql/chain_list_all.sql to fetch all chains in the market
   * uses sql/delete_existing_records.sql to delete all existing chain/tags mapping
2. The second DAG is for the incremental load, triggered automatically on daily basis, fetches untagged chains and tags them.
   * Uses sql/chain_list_new.sql to fetch all chains that are not in the dim_chain_tags table

The following figure illustrates the DAGs' flow:

![dag_workflow.jpg](images%2Fdag_workflow.jpg)

### 4. Saving results
Results are saved to below tables

1. **data_tlabel.dim_tags**
- **tag_id**:  a combination of string and int, ex: geo_123, dish_456, etc…
- **tag_name**: ex: “American”, “Pasta”, etc..
- **category**: ex: ‘geographical’, ‘dish’, etc…
- **country_code**: ex:  ‘AE’, ‘KW’, etc…
- **dwh_entry_timestamp**: entry date and time

2. **data_tlabel.dim_chain_tags**
- **chain_id**: same chain_id as in dim_chain
- **tag_id**: same as in dim_tags
- **explanation**: Brief explanation behind the tag choice
- **dwh_entry_timestamp**: entry date and time

### 5. MainDB: 
Move data to the main DB to reflect in the app.
<br> This part hasn't been discussed yet, but it should be handled with reverse ETL from BQ to the main DB.

## Notes:
1. Running the code is straightforward, everything starts with **predict.py** that takes the following external arguments:
```
  --category CATEGORY               Category for prediction
  --country_code COUNTRY_CODE       Country code
  --batch_size BATCH_SIZE           Batch size for processing chains
  --overwrite                       Whether to overwrite existing data
```
2. The LLM token is stored as a secret in the GCP secret manager <projects/tlb-data-prod/secrets/tlabel_openai/versions/latest> and fetched at runtime
3. To run the code locally, either to get read access to GCP secret manager or to use a new token from [LLM proxy](https://data.talabat.com/apps/genaiproxy/Tokens)

## Future work
1. Build a robust evaluation pipeline to measure the performance if the prompt message changes
2. Implement a reverse ETL to move the data to the main DB
3. Implement a tag deletion mechanism, that deletes the tag from dim_tags, drop tagged chains from dim_chain_tags, and re-tag them
4. Implement a manual tagging mechanism to allow the business to manually tag chains
