DELETE FROM `{{ dim_chain_tags }}`
WHERE
    chain_id IN ({{ chain_ids }})
  AND (
      tag_id LIKE "{{ tag_prefix }}%"
      OR tag_id IS NULL
    )
