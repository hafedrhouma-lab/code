import datetime

def write_query(start_date=None, end_date=None, sku_pattern=None, warehouse_pattern=None):

    sku_condition = f"AND dwh_dmart_pv2.sku LIKE '{sku_pattern}%' " if sku_pattern else ""
    warehouse_condition = f"AND dpd_w.warehouse_id LIKE '{warehouse_pattern}%' " if warehouse_pattern else ""

    query =f'''
    WITH
        sku_info AS(
        select
            dwh_dmart_pv2.country_code,
            dwh_dmart_pv2.sku,
            dpd_w.warehouse_id,
            V.platform_vendor_id,
            category_parent_english AS parent_category,
            master_category_english AS master_category
        from
            `fulfillment-dwh-production.curated_data_shared_dmart.products_v2` dwh_dmart_pv2,
            UNNEST(warehouse_info) W,
            UNNEST(vendor_info) V
        INNER JOIN
        (
        SELECT
            DISTINCT warehouse_id,
            global_entity_id
        FROM
            `tlb-data-prod.data_platform_dmart.warehouses`
        ) dpd_w
            ON dpd_w.warehouse_id = w.warehouse_id
            AND dpd_w.global_entity_id = dwh_dmart_pv2.global_entity_id
        WHERE 1=1
            {sku_condition}
            {warehouse_condition}
        ),



    orders AS (
        SELECT
            dp_foit.order_date,
            dp_di.product_sku AS sku,
            dp_foin.vendor_id,
            dp_foin.country_code AS country_code,
            sum(sold_quantity) as sales,
            sum(ordered_quantity),
            sum(delivered_quantity),
            avg(dp_foit.order_item_price_lc) as order_item_price_lc

        FROM
            `tlb-data-prod.data_platform.fct_order_info` dp_foin
        LEFT JOIN
            `tlb-data-prod.data_platform.fct_order_item` dp_foit
        ON
            dp_foin.order_id = dp_foit.order_id
            AND dp_foit.order_date >= '{start_date}'
            AND dp_foit.order_date <= '{end_date}'
            AND dp_foin.is_successful
            LEFT JOIN `tlb-data-prod.data_platform.dim_item` dp_di
        ON dp_di.item_id = dp_foit.item_id
        WHERE
            dp_foin.order_date <= '{end_date}'
            AND dp_foin.order_date  >= '{start_date}'
            AND dp_foin.is_successful
            AND dp_foin.is_darkstore
            AND dp_di.vertical_class <> 'food'
            AND dp_foin.country_code='KW'
        GROUP BY
            1,
            2,
            3,
            4
    ),
    merged_table AS(SELECT
                orders.order_date,
                orders.order_item_price_lc,
                sku_info.country_code,
                sku_info.sku,
                sku_info.warehouse_id,
                sku_info.parent_category,
                sku_info.master_category,
                orders.sales
            FROM
                sku_info
            JOIN
                orders
            ON
            orders.sku = sku_info.sku
            AND
            CAST(orders.vendor_id as STRING) = sku_info.platform_vendor_id)

    SELECT
    order_date,
    country_code,
    sku,
    warehouse_id,
    order_item_price_lc,
    parent_category,
    master_category,
    sales,
    FROM(
        SELECT
        merged_table.order_date,
        merged_table.country_code,
        merged_table.sku,
        merged_table.warehouse_id,
        merged_table.parent_category,
        merged_table.master_category,
        merged_table.sales,
        merged_table.order_item_price_lc
        from
        merged_table )
    '''
    return query
