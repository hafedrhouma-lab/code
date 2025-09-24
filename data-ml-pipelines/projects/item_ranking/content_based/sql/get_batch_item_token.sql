SELECT
  dim_item.source_system_item_id,
  dim_chain.chain_name,
  dim_chain.country_code,
  LOWER(
    REGEXP_REPLACE(
      CONCAT(
        IFNULL(dim_item.item_name_en, ''), ' ',
        IFNULL(dim_item.item_description_en, ''), ' ',
        IFNULL(ese_items.chocies_item_names, '')
      ),
      r'[^a-zA-Z0-9\s]', '' -- Keep only alphanumeric characters and spaces
    )
  ) AS tokens

FROM `tlb-data-prod.data_platform.dim_item` dim_item

INNER JOIN `tlb-data-prod.data_platform_catalogue_db.menu_section` menu_section
ON menu_section.id = dim_item.menu_section_id

INNER JOIN `tlb-data-prod.data_platform.dim_chain` dim_chain
ON dim_item.chain_id = dim_chain.chain_id

LEFT JOIN `tlb-data-prod.data_platform.ese_items` ese_items
ON ese_items.parent_item_id = dim_item.item_id

WHERE 1=1
  AND menu_section.name = "M41"
  AND item_type_name = "main"
  AND dim_item.is_active
  AND NOT dim_item.is_deleted
  AND dim_item.vertical = 'food'
  AND dim_chain.country_code = '{{ country_code }}'
  AND dim_item.source_system_item_id IN ({{ batch_item_ids }})
