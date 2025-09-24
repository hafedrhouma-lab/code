SELECT  DISTINCT(account_id)
FROM `tlb-data-prod.data_platform.ese_orders_token`
WHERE TIMESTAMP(order_time) >= TIMESTAMP(CURRENT_DATE() - {{ days_lag }})
AND MOD(account_id, 10) = {{ last_digit }};
