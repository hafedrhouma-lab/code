CREATE TABLE inference_features_for_menu_two_tower_v1_per_account
(
    account_id       integer,
    country_code     character varying,
    freq_items       character varying,
    freq_items_names character varying,
    prev_items       character varying,
    prev_items_names character varying
);

CREATE TABLE inference_features_for_menu_two_tower_v1_per_account_per_chain
(
    account_id             integer,
    country_code           character varying,
    chain_id               integer,
    freq_items             character varying,
    freq_items_names       character varying,
    prev_items             character varying,
    prev_items_names       character varying,
    chain_prev_items       character varying,
    chain_prev_items_names character varying
);


INSERT INTO inference_features_for_menu_two_tower_v1_per_account
    (account_id, country_code, freq_items, freq_items_names, prev_items, prev_items_names)
VALUES
(-1, 'EG', 'item1 item2 item3', 'item1_name item2_name item3_name', 'prev_item1 prev_item2', 'prev_item1_name prev_item2_name');

INSERT INTO inference_features_for_menu_two_tower_v1_per_account_per_chain
    (account_id, country_code, chain_id,
     freq_items, freq_items_names, prev_items, prev_items_names,
     chain_prev_items, chain_prev_items_names)
VALUES
(10914736, 'EG', 665933,
 'item1 item2 item3', 'item1_name item2_name item3_name', 'prev_item1 prev_item2', 'prev_item1_name prev_item2_name',
 'chain_prev_item1 chain_prev_item2', 'chain_prev_item1_name chain_prev_item2_name');
