WITH
orders AS (
    SELECT account_id, chain_name, item_name, item_description, order_rank
    FROM `{{ order_table }}`
    WHERE 1=1
        AND account_id IN ({{ batch_account_ids }})
        AND order_rank <= {{ max_order_rank }}
    ORDER BY order_time DESC
),
keywords AS (
    SELECT account_id, search_history, search_rank
    FROM `{{ search_table }}`
    WHERE 1=1
        AND account_id IN ({{ batch_account_ids }})
        AND search_rank <= {{ max_order_rank }}
    ORDER BY search_time DESC
)
-- Tokenization on the fly
SELECT
    COALESCE(orders.account_id, keywords.account_id) AS account_id,
    ARRAY_TO_STRING(
        [
            STRING_AGG(
                CONCAT(orders.chain_name, ' ', orders.item_name, ' ', orders.item_description),
                ' '
                ORDER BY orders.order_rank
            ),
            STRING_AGG(
                keywords.search_history,
                ' '
                ORDER BY keywords.search_rank
            )
        ], ' '
    ) AS tokens
FROM
    orders
    FULL OUTER JOIN keywords
    ON orders.account_id = keywords.account_id
    AND orders.order_rank = keywords.search_rank
GROUP BY account_id;
