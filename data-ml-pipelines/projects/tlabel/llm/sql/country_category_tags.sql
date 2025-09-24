SELECT *
FROM `{{ dim_tags_table }}`
WHERE country_code = "{{ country }}"
  AND category = "{{ category }}"