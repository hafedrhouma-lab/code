-- Step 1: Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS `{{ output_table }}` (
  source_system_item_id STRING,
  country_code STRING,
  transformer_embeddings ARRAY<FLOAT64>,
  feature_timestamp TIMESTAMP
);

-- Step 2: Query unique source_system_item_id from dim_item
--         that do not exist in the above table
SELECT DISTINCT di.source_system_item_id
FROM `tlb-data-prod.data_platform.dim_item` AS di
INNER JOIN `tlb-data-prod.data_platform_catalogue_db.menu_section` AS ms
  ON ms.id = di.menu_section_id
INNER JOIN `tlb-data-prod.data_platform.dim_chain` AS dc
  ON di.chain_id = dc.chain_id
WHERE
    ms.name = "M41"
    AND di.item_type_name = "main"
    AND di.is_active
    AND NOT di.is_deleted
    AND di.vertical = 'food'
    AND UPPER(dc.country_code) = '{{ country_code }}'
    -- Only return those that do NOT exist in item_content_based_embeddings
    AND di.source_system_item_id NOT IN (
      SELECT source_system_item_id
      FROM `{{ output_table }}`
    );
