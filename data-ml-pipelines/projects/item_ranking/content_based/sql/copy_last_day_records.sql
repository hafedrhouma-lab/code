INSERT INTO `{{ output_table }}` (account_id, transformer_embeddings, feature_timestamp)
SELECT
  account_id,
  transformer_embeddings,
  TIMESTAMP(CURRENT_DATE())
FROM `{{ output_table }}`
WHERE
    feature_timestamp = (SELECT MAX(feature_timestamp) FROM `{{ output_table }}`)
    AND MOD(account_id, 10) = {{ last_digit }}
    -- Only insert if the record does not exist for today, to avoid duplication id DAG re-runs
    AND NOT EXISTS (
            SELECT 1
            FROM `{{ output_table }}`
            WHERE DATE(feature_timestamp) = CURRENT_DATE()
        );
