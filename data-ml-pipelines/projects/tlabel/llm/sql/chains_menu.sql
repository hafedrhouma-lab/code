SELECT
      dc.chain_id,
      dc.chain_name,
      ei.parent_item_category_name,
      ei.parent_item_name_en,
      ei.parent_item_description_en,
      ROUND(ei.parent_orders_percentage * 100, 2) AS parent_orders_percentage
FROM
    `tlb-data-prod.data_platform.ese_items` ei
  INNER JOIN
    `tlb-data-prod.data_platform.dim_chain` dc
    ON ei.chain_id = dc.chain_id
WHERE 1=1
    AND dc.country_code = "{{ country_code }}"
    AND dc.chain_status < 4
    AND dc.chain_id IN ({{ batch_list }})
    AND ei.is_active

ORDER BY
    parent_orders_percentage DESC
