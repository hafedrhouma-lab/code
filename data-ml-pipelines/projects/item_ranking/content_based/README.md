# Item Ranking

## Contents
1. [Content-based item ranking](#content-based-item-ranking)
   1. [DAGs](#dags)
      1. [Account Embeddings Generation](#account-embeddings-generation)
         1. [Initial Load](#initial-load)
         2. [Incremental Load](#incremental-load)
      2. [Item Embeddings Generation](#item-embeddings-generation)
         1. [Incremental Load](#incremental-load)
      3. [Notes](#notes)
      
# Content-based item ranking
## 1. DAGs
### 1.1 Account Embeddings Generation
Account embeddings generation happens as an initial load that happens once, then incremental load that keeps running on daily basis.
Both stages are triggered using the *generate_account_embeddings.py* script, and illustrated by the following flowchart
The difference between initial and incremental loads is just what input arguments we give to the script

![account_embeddings.jpg](images%2Faccount_embeddings.jpg)

#### 1.1.1 Initial Load
Consists of 10 parallel tasks, each handling accounts ending with a certain digit (0-9)
- **--last_digit:** parameter is set to 0-9, to specify which digit to handle in each task
- **--days_lag:** parameters is set to 180, meaning we generate embeddings for active users in the last 180 days
- **--batch_size:** is set to 500, meaning we encode 500 accounts ata time and upload them to bigquery before we process the next batch
- **--max_order_rank:** controls the number of previous orders and keyword searches to consider and encode into the embeddings. 15 tends to be a good value and suitable to the model's input (512 tokens)
- **--float_precision:** controls the number of decimal places to keep in the embeddings. 8 is a good value to keep the embeddings small and efficient

#### 1.1.2 Incremental Load
Same as the initial load, but with different input arguments
- **--days_lag:** is set to 1, meaning we generate embeddings for active users in the last 1 day


### 1.2 Item Embeddings Generation
Unlike account embeddings, item embeddings are generated in a single incremental task, as items don't have history as such and aren't supposed to change with time.
There is only one DAG and is triggered using the *generate_item_embeddings.py* script, followed by the following arguments

![item_embeddings.jpg](images%2Fitem_embeddings.jpg)

#### 1.2.1 Incremental Load
- **--country_code:** specifies what country to generate item embeddings for
- **--last_digit:** parameter is set to 0-9, to specify which digit to handle in each task
- **--batch_size:** is set to 500, meaning we encode 500 items at a time and upload them to bigquery before we process the next batch
- **--float_precision:** controls the number of decimal places to keep in the embeddings. 8 is a good value to keep the embeddings small and efficient

### 1.3 Notes
1. Implications of skipping the generation of incremental account embeddings
   1. we don't get updated embeddings on that day
   2. New users are missed
   3. The table will have gaps in days
   4. The next successful run will get a copy of the last know table and continue from there
2. Implications of skipping the generation of incremental item embeddings
   1. New items are missed on this day
   2. The next successful run will recover everything.