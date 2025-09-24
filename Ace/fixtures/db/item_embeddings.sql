CREATE EXTENSION IF NOT EXISTS vector;

drop table if exists item_embeddings;
create table item_embeddings
(
    item_id                 bigint,
    item_name_en            text,
    chain_id                int,
    chain_name              text,
    embedding               vector(384),
    order_count             int,
    unique_order_dates      int,
    vertical                varchar(100),
    item_price_average      float,
    is_promotional_category boolean

);

create table if not exists customers_chains_items
(
    customer_id bigint,
    all_chains  int[],
    all_items   bigint[],
    update_date date
);

create index if not exists customers_chains_items_customer_id_idx
    on customers_chains_items (customer_id);



drop function if exists f_items_semantic_search;
create or replace function f_items_semantic_search(p_country_code varchar, p_verticals character varying[],
                                                   p_chains integer[],
                                                   p_embedding vector,
                                                   p_limit integer,
                                                   p_offset integer,
                                                   p_comparison_sign character varying,
                                                   p_price double precision,
                                                   p_past_orders_only boolean DEFAULT NULL::boolean,
                                                   p_customer_id bigint DEFAULT NULL::bigint)
    returns TABLE
            (
                item_id                    bigint,
                content                    text,
                chain_id                   integer,
                chain_name                 text,
                order_count                integer,
                unique_order_dates         integer,
                vertical                   character varying,
                avg_original_item_price_lc double precision,
                is_promotional_category    boolean,
                distance                   double precision
            )
    language plpgsql
as
$$
    # variable_conflict use_column
begin
    SET LOCAL ivfflat.probes = 35;
    return query
        select item_id,
               content,
               chain_id,
               chain_name,
               order_count,
               unique_order_dates,
               vertical,
               avg_original_item_price_lc,
               is_promotional_category,
               embedding <#> p_embedding as distance
        from all_items_embeddings_metadata
        where country_code = p_country_code
          and order_count >= 30
          and embedding <#> p_embedding < -0.4
          and case when array_length(p_chains, 1) > 0 then chain_id = any (p_chains) else true end
          and case when array_length(p_verticals, 1) > 0 then vertical = any (p_verticals) else true end
          and case
                  when p_comparison_sign = '>' then avg_original_item_price_lc > p_price
                  when p_comparison_sign = '<' then avg_original_item_price_lc < p_price
                  when p_comparison_sign = '>=' then avg_original_item_price_lc >= p_price
                  when p_comparison_sign = '<=' then avg_original_item_price_lc <= p_price
                  when p_comparison_sign = '=' then avg_original_item_price_lc = p_price
                  else true
            end
          and case
                  when p_past_orders_only = true then item_id = any ((select unnest(all_items)
                                                                      from customers_chains_items
                                                                      where customer_id = p_customer_id))
                  when p_past_orders_only = false then item_id != any ((select unnest(all_items)
                                                                        from customers_chains_items
                                                                        where customer_id = p_customer_id))
                  else true
            end
        order by embedding <#> p_embedding
        limit p_limit
        offset p_offset
    ;
end;
$$;
