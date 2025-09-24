DELETE FROM `{{ output_table }}`
WHERE
    account_id IN ({{ account_id_list }})
    AND feature_timestamp = TIMESTAMP(CURRENT_DATE());