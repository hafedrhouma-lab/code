SELECT count(distinct(account_id)) count_distinct, count(account_id) count
FROM `{{ output_table }}`
WHERE DATE(feature_timestamp) = CURRENT_DATE()
AND MOD(account_id, 10) = {{ last_digit }};