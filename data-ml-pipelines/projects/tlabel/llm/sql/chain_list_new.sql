SELECT DISTINCT chain_id
FROM `tlb-data-prod.data_platform.dim_chain`
WHERE 1=1
    AND country_code = "{{ country_code }}"
    AND chain_status < 4

    AND chain_id NOT IN (
        SELECT
            DISTINCT CAST(chain_id as INT)
        FROM
            `{{dim_chain_tags}}` dct
        INNER JOIN `{{dim_tags}}` dt
            ON dct.tag_id = dt.tag_id
        WHERE
            dt.country_code = "{{ country_code }}" AND
            dt.category = "{{ category }}"
    )

ORDER BY chain_id