# SQL Queries


## Table Creation
### Session Searches Aggregated

The table used to generate the aggregated performances by filter in the next section (Searches Aggregated Query)

<details>
  <summary>Show query</summary>

```sql
CREATE TABLE temp_hafed.queries_performances_session_level
 (
    DATE DATE,
    session_id STRING,
    user_selected_country STRING,
    user_selected_city STRING,
    area_name STRING,
    OS STRING,
    search_query_lang STRING,
    search_type STRING,
    query_entered STRING,
    number_search NUMERIC,
    search_zero_results NUMERIC,
    number_search_non_zero_results NUMERIC,
    avg_search_results_count NUMERIC,
    search_result_clicked BOOLEAN,
    placed_order BOOLEAN,
    session_placed_order BOOLEAN
 )
 PARTITION BY DATE
 CLUSTER BY
   user_selected_country, 
   user_selected_city, 
   area_name
 OPTIONS (
   description="a table showing text queries performances for grocery"
 )
AS 
WITH session_query AS (
    SELECT 
        DATE,
        session_id,
        user_selected_country,
        user_selected_city,
        location_info.area_name,
        operating_system AS OS,
        CASE
            WHEN ifnull(regexp_extract(search_query_entered , r'([\p{Arabic}]+)'), '') != '' THEN 'arabic'
            WHEN ifnull(regexp_extract(search_query_entered , r'([\p{Latin}]+)'), '') != '' THEN 'english'
            ELSE 'unknown'
        END AS search_query_lang,
        CASE 
            WHEN search_results_tab NOT LIKE ('%item%') THEN 'store_search' 
            ELSE 'item_search' 
        END AS search_type,
        LOWER(search_query_entered) AS query_entered,
        COUNT(*) AS number_search,
        --serch relevance
        SUM(CASE WHEN (search_results_count = 0 OR search_results_count IS NULL) THEN 1 ELSE 0 END) AS search_zero_results,
        SUM(CASE WHEN search_results_count > 0 THEN 1 ELSE 0 END) AS search_non_zero_results,
        CAST(AVG(CASE WHEN search_results_count > 0 THEN search_results_count END) AS INT) AS avg_search_results_count,
        --search efficiency
        logical_or(
            CASE
                WHEN viewed_menu THEN TRUE
                ELSE FALSE
            END) AS search_result_clicked,
        logical_or(
        CASE
                WHEN placed_order THEN TRUE
                ELSE FALSE
        END) AS placed_order,
    FROM 
        `bta---talabat.data_platform.fct_sub_session_search`
    INNER JOIN 
        `bta---talabat.data_platform.dim_location_info` location_info
    ON 
        user_selected_area=cast(location_info.area_id as string)
    WHERE 
        DATE >=CURRENT_DATE()-120
    -- clean areas
    AND
        user_selected_country IS NOT NULL 
    AND 
        user_selected_city IS NOT NULL 
    AND 
        user_selected_area IS NOT NULL
    -- vertical
    AND
        (search_results_tab IN ('groceries_stores','groceries_items','groceries') 
        OR 
        (search_results_tab='items' AND vertical='grocery'))
    -- clean queries
    AND 
        LENGTH(search_query_entered)>0
    AND 
        LOWER(trim(search_query_entered)) NOT IN ('na','')
    -- when found_via* is NULL, no click or placed order infos
    GROUP BY 
        1,2,3,4,5,6,7,8,9
)
SELECT 
session_query.*,
session_vertical.placed_order AS session_placed_order
FROM 
session_query
LEFT JOIN
`bta---talabat.data_platform.fct_session_vertical` session_vertical
USING
(session_id)
WHERE
    session_vertical.vertical = 'grocery'
```
</details>


### In Vendor Search

The table used to generate the aggregated performances by filter in the next section (In vendor search)

<details>
  <summary>Show query</summary>

```sql
CREATE TABLE temp_hafed.queries_performances_in_vendor_search
 (
    DATE DATE,
    store_type STRING,
    chain_name_en STRING,
    query_entered STRING,
    search_query_lang STRING,
    os STRING,
    user_selected_country STRING,
    user_selected_city STRING,
    area_name STRING,
    number_search NUMERIC,
    number_sessions NUMERIC,
    number_search_atc NUMERIC,
    number_search_transaction NUMERIC,
    number_session_atc NUMERIC,
    number_session_buying NUMERIC,
    nb_searches_zr NUMERIC
 )
 PARTITION BY DATE
 CLUSTER BY
   user_selected_country, 
   user_selected_city, 
   area_name
 OPTIONS (
   description="a table showing text queries performances for grocery in vendor search"
 )
AS 
    WITH 
    
    customer_future_path_table AS (


        SELECT
        session_id,
        DATE,
        sub_vertical AS store_type,
        vendor.chain_name_en,
        LOWER(fct_hit.search_term) AS query_entered,
        CASE
            WHEN ifnull(regexp_extract(fct_hit.search_term , r'([\p{Arabic}]+)'), '') != '' 
            THEN 'arabic'
            WHEN ifnull(regexp_extract(fct_hit.search_term , r'([\p{Latin}]+)'), '') != '' 
            THEN 'english'
        ELSE 'unknown'
        END AS search_query_lang,
        device.operating_system AS os,
        location.country_name AS user_selected_country,
        location.city_name AS user_selected_city,
        location.area_name,
        event_action.event_action,
        items_number,
        
        STRING_AGG
            (
            CASE
                WHEN event_action.event_action = 'product_choice.opened' THEN 'p'
                WHEN event_action.event_action = 'shop_details.loaded' THEN 'v'
                WHEN event_action.event_action = 'search_results.loaded' THEN 'sl'
                WHEN event_action.event_action = 'category_details.loaded' THEN 'c'
                WHEN event_action.event_action = 'checkout.loaded' THEN 'co'
                WHEN event_action.event_action = 'transaction' THEN 't'
                WHEN event_action.event_action = 'add_cart.clicked' THEN 'a'
            END
            ,'-'
            ) 
        OVER 
            (
            PARTITION BY fct_hit.session_id
            ORDER BY hit_number
            ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            )
        AS 
            customer_future_path

        FROM 
            `bta---talabat.data_platform.fct_hit` fct_hit
        LEFT JOIN
            `bta---talabat.data_platform.dim_session_event` event_action 
        ON
            fct_hit.session_event_id=event_action.session_event_id
        LEFT JOIN
            `bta---talabat.data_platform.dim_vendor` AS vendor
        ON
            fct_hit.vendor_id=vendor.vendor_id
        LEFT JOIN 
            `bta---talabat.data_platform.dim_device` device
        ON 
            fct_hit.device_id=device.device_id
        LEFT JOIN
            `bta---talabat.data_platform.dim_location` AS location
        ON
            fct_hit.location_id=location.location_id 
        WHERE
            date >= CURRENT_DATE()-180
        AND
            vendor.is_grocery
        AND 
            event_action.event_action IN 
                (
                'shop_details.loaded',
                'product_choice.opened',
                'search_results.loaded',
                'category_details.loaded',
                'add_cart.clicked',
                'checkout.loaded',
                'transaction'
                )


    ),

    customer_future_path_labelling AS (


        SELECT 
        path.*,
        REGEXP_CONTAINS(path.customer_future_path,'^(p-)*a') AS atc,
        REGEXP_CONTAINS(path.customer_future_path,'t$') AS transaction,
        FROM 
        customer_future_path_table path
        WHERE
        event_action = 'search_results.loaded'


    ),


    aggregate_table AS (


        SELECT 
        DATE,
        store_type,
        chain_name_en,
        query_entered,
        search_query_lang,
        os,
        user_selected_country,
        user_selected_city,
        area_name,
        COUNT(*) AS number_search,
        COUNT(DISTINCT session_id) AS number_sessions,
        SUM
        (
            CASE 
            WHEN atc THEN 1
            ELSE 0
            END
        ) 
        AS number_search_atc,
        SUM
        (
            CASE 
            WHEN transaction THEN 1
            ELSE 0
            END
        ) 
        AS number_search_transaction,
        COUNT
        (
            DISTINCT
            CASE
                WHEN atc 
                THEN session_id
            END
        ) number_session_atc,
        COUNT
        (
            DISTINCT
            CASE
                WHEN transaction 
                THEN session_id
            END
        ) number_session_buying,
        SUM
        (
            CASE 
            WHEN items_number='0' 
            THEN 1 
            ELSE 0 END
        ) nb_searches_zr
        FROM  
        customer_future_path_labelling
        GROUP BY 1,2,3,4,5,6,7,8,9
        

    )

    SELECT
        *
    FROM aggregate_table

```
</details>

## Searches Aggregated

### Homepage Search

Query used to serve as dataset for homepage search Queries clustering. It is using  [Jinja package](https://pypi.org/project/Jinja2/)

<details>
  <summary>Show query</summary>

```sql
--aggregate are precomputed in created table

--aggregate are precomputed in created table

SELECT
  query_entered,
  COUNT(DISTINCT session_id) AS number_sessions,
  SUM(
    CASE
        WHEN search_result_clicked IS TRUE THEN 1
        ELSE 0 END
     )
     AS number_sessions_that_clicked,
  SUM(
    CASE
        WHEN placed_order IS TRUE THEN 1
        ELSE 0 END
     )
     AS number_query_sessions_that_placed_order,
  SUM(
    CASE
        WHEN session_placed_order IS TRUE THEN 1
        ELSE 0 END
     )
     AS number_sessions_that_placed_order,
  SUM(number_search) AS number_search,
  SUM(search_zero_results) AS search_zero_results,
  CAST(
    AVG(avg_search_results_count) AS INT
    )
    AS avg_search_results_count
FROM
  `bta---talabat.temp_hafed.queries_performances_session_level`
WHERE
    date >= '{{user_selected_values.get("date_start")}}'
AND
    user_selected_country='{{user_selected_values.get("country")}}'
{% if user_selected_values.get("meta").get("city") %}
AND
    user_selected_city='{{user_selected_values.get("meta").get("city")}}'
{% endif %}
{% if user_selected_values.get("meta").get("area") %}
AND
    area_name='{{user_selected_values.get("meta").get("area")}}'
{% endif %}
AND
    (
    {%- for os in user_selected_values.get("os") %}
    OS='{{os}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
AND
    (
    {%- for language in user_selected_values.get("language") %}
    search_query_lang='{{language}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
AND
    (
    {%- for search_type in user_selected_values.get("search_type") %}
    search_type='{{search_type}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
GROUP BY
  query_entered

```

</details>


### In Vendor Search

Query used to serve as dataset for In Vendor Search Queries Clustering. It is using  [Jinja package](https://pypi.org/project/Jinja2/)

<details>
  <summary>Show query</summary>

```sql
SELECT
    query_entered,
    SUM(number_sessions) AS number_sessions,
    SUM(number_search) AS number_search,
    SUM(number_session_atc)/SUM(number_sessions)*100 AS ATC_percentage,
    SUM(number_session_buying)/SUM(number_sessions)*100 AS CVR_percentage,
    SUM(nb_searches_zr)/SUM(number_search)*100 AS ZRR_percentage
FROM
    `bta---talabat.temp_hafed.queries_performances_in_vendor_search`
WHERE
    date >= '{{user_selected_values.get("date_start")}}'
AND
    user_selected_country='{{user_selected_values.get("country")}}'
{% if user_selected_values.get("meta").get("city") %}
AND
    user_selected_city='{{user_selected_values.get("meta").get("city")}}'
{% endif %}
{% if user_selected_values.get("meta").get("area") %}
AND
    area_name='{{user_selected_values.get("meta").get("area")}}'
{% endif %}
{% if user_selected_values.get("meta").get("store_name") %}
AND
    chain_name_en='{{user_selected_values.get("meta").get("store_name")}}'
{% endif %}
AND
    (
    {%- for os in user_selected_values.get("os") %}
    OS='{{os}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
AND
    (
    {%- for language in user_selected_values.get("language") %}
    search_query_lang='{{language}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
AND
    (
    {%- for store_type in user_selected_values.get("store_type") %}
    store_type='{{store_type}}'
    {%- if not loop.last -%}
        OR
    {%- endif -%}
    {%- endfor %}
    )
AND
    LENGTH(query_entered)>0
AND
    LOWER(trim(query_entered)) NOT IN ('na','')
GROUP BY
  query_entered
```
</details>
Args:

* ```date_start```: date from which we want queries
* ```user_selected_country```: country to query searches
* ```user_selected_city```: city to query searches
* ```area_name```: area to query search
* ```OS```: user's OS for query search
* ```search_query_lang```: user's language for query search
* ```search_type```: user's selected tab for query search

