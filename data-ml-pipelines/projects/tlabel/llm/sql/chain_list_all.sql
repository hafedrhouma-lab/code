SELECT DISTINCT chain_id
FROM `tlb-data-prod.data_platform.dim_chain`
WHERE 1=1
    AND country_code = "{{ country_code }}"
    AND chain_status < 4
ORDER BY chain_id

