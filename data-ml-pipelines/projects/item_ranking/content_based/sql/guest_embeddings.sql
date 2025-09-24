INSERT INTO `{{ output_table }}`
WITH relevant_accounts AS (
  -- 1) Find all (account_id, country_id, country_code) pairs from the last {{ lag_days }} days
  SELECT DISTINCT
    eot.account_id,
    dci.country_id,
    foi.country_code
  FROM `tlb-data-prod.data_platform.ese_orders_token` AS eot
  JOIN `tlb-data-prod.data_platform.fct_order_info` AS foi
    ON eot.order_id = foi.order_id
    AND eot.account_id = foi.account_id
  JOIN `tlb-data-prod.data_platform.dim_country_info` AS dci
    ON foi.country_code = dci.country_code
  WHERE TIMESTAMP(eot.order_time) >= TIMESTAMP(CURRENT_DATE() - {{ days_lag }})
),
relevant_embeddings AS (
  -- 2) Join the above accounts to your embeddings table
  SELECT
    ra.country_id,
    ra.country_code,
    emb.transformer_embeddings
  FROM relevant_accounts AS ra
  JOIN `{{ output_table }}` AS emb
    ON ra.account_id = emb.account_id
  WHERE emb.feature_timestamp = (SELECT MAX(feature_timestamp) FROM `{{ output_table }}`)
),
flattened AS (
  -- 3) Flatten each array element, while keeping track of its dimension index
  SELECT
    country_id,
    country_code,
    OFFSET AS dim_index,
    val AS embedding_value
  FROM relevant_embeddings,
       UNNEST(transformer_embeddings) AS val WITH OFFSET
),
dim_averages AS (
  -- 4) Compute the dimension-wise average per country
  SELECT
    country_id,
    country_code,
    dim_index,
    AVG(embedding_value) AS dimension_avg
  FROM flattened
  GROUP BY country_id, country_code, dim_index
)
-- 5) Reconstruct each country's embedding vector and insert it
SELECT
  -country_id AS account_id,  -- negative of the country's ID
  ARRAY_AGG(dimension_avg ORDER BY dim_index) AS transformer_embeddings,
  TIMESTAMP(CURRENT_DATE()) AS feature_timestamp
FROM dim_averages
GROUP BY country_id, country_code;
