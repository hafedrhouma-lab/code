SELECT count(distinct(source_system_item_id)) count_distinct, count(source_system_item_id) count
FROM {{ output_table }}
WHERE DATE(feature_timestamp) = CURRENT_DATE()
AND country_code = "{{ country_code }}"
