def get_predict_actual_query():

    query = '''
    DECLARE start_date DATE;
    DECLARE end_date DATE;
    
    -- Get the start_date and end_date from the forecast table
    SET start_date = DATE_SUB((SELECT MIN(forecast_date) FROM `data_platform_sku_forecasting.tft_forecast`), INTERVAL 14 DAY);
    SET end_date = (SELECT MAX(forecast_date) FROM `data_platform_sku_forecasting.tft_forecast`);
    
    WITH
        sku_info AS (
            SELECT
                dwh_dmart_pv2.country_code,
                dwh_dmart_pv2.sku,
                dpd_w.warehouse_id,
                V.platform_vendor_id
            FROM
                `fulfillment-dwh-production.curated_data_shared_dmart.products_v2` dwh_dmart_pv2,
                UNNEST(warehouse_info) W,
                UNNEST(vendor_info) V
            INNER JOIN (
                SELECT
                    DISTINCT warehouse_id,
                    global_entity_id
                FROM
                    `tlb-data-prod.data_platform_dmart.warehouses`
            ) dpd_w
            ON dpd_w.warehouse_id = w.warehouse_id
            AND dpd_w.global_entity_id = dwh_dmart_pv2.global_entity_id
        ),
    
        orders AS (
            SELECT
                dp_foit.order_date,
                dp_di.product_sku AS sku,
                dp_foin.vendor_id,
                dp_foin.country_code AS country_code,
                SUM(sold_quantity) AS sales,
                SUM(ordered_quantity),
                SUM(delivered_quantity),
                AVG(dp_foit.order_item_price_lc) AS order_item_price_lc
            FROM
                `tlb-data-prod.data_platform.fct_order_info` dp_foin
            LEFT JOIN `tlb-data-prod.data_platform.fct_order_item` dp_foit
            ON dp_foin.order_id = dp_foit.order_id
            AND dp_foit.order_date >= start_date
            AND dp_foit.order_date <= end_date
            AND dp_foin.is_successful
            LEFT JOIN `tlb-data-prod.data_platform.dim_item` dp_di
            ON dp_di.item_id = dp_foit.item_id
            WHERE
                dp_foin.order_date <= end_date
                AND dp_foin.order_date >= start_date
                AND dp_foin.is_successful
                AND dp_foin.is_darkstore
                AND dp_di.vertical_class <> 'food'
                AND dp_foin.country_code = 'KW'
            GROUP BY
                1, 2, 3, 4
        ),
    
        merged_table AS (
            SELECT
                orders.order_date,
                sku_info.country_code,
                sku_info.sku,
                sku_info.warehouse_id,
                orders.sales
            FROM
                sku_info
            JOIN
                orders
            ON orders.sku = sku_info.sku
            AND CAST(orders.vendor_id AS STRING) = sku_info.platform_vendor_id
        ),
    
        valid_skus_warehouses AS (
            SELECT DISTINCT
                sku,
                warehouse_id
            FROM
                `data_platform_sku_forecasting.tft_forecast`
        )
    
    SELECT
        COALESCE(ac.order_date, fc.forecast_date) AS date,
        COALESCE(ac.country_code, fc.country_code) AS country_code,
        COALESCE(ac.sku, fc.sku) AS sku,
        COALESCE(ac.warehouse_id, fc.warehouse_id) AS warehouse_id,
        ac.sales,
        fc.*
    FROM (
        SELECT
            merged_table.order_date,
            merged_table.country_code,
            merged_table.sku,
            merged_table.warehouse_id,
            merged_table.sales
        FROM
            merged_table
        JOIN
            valid_skus_warehouses
        ON merged_table.sku = valid_skus_warehouses.sku
        AND merged_table.warehouse_id = valid_skus_warehouses.warehouse_id
    ) ac
    FULL OUTER JOIN `data_platform_sku_forecasting.tft_forecast` fc
    ON CAST(ac.order_date AS DATE) = fc.forecast_date
    AND ac.warehouse_id = fc.warehouse_id
    AND ac.sku = fc.sku;
    '''
    return query