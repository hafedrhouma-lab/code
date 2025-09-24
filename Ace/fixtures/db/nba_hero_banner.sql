-- auto-generated definition
create table if not exists ipm_rule_based_predictions
(
    account_id  integer not null
        constraint ipm_rule_based_predictions_pkey
            primary key,
    loaded_date date,
    morning     integer[],
    afternoon   integer[],
    evening     integer[],
    midnight    integer[]
);

insert into ipm_rule_based_predictions (account_id, loaded_date, morning, afternoon, evening, midnight)
values (1, '2023-03-18', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (2, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (9155336, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3836513, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (18610653, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (14138382, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (28884781, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (8255293, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (4261247, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}');

-- auto-generated definition
create table if not exists ipm_guided_rl_predictions
(
    account_id  integer not null
        constraint ipm_guided_rl_predictions_pkey
            primary key,
    loaded_date date,
    morning     integer[],
    afternoon   integer[],
    evening     integer[],
    midnight    integer[]
);

insert into ipm_guided_rl_predictions (account_id, loaded_date, morning, afternoon, evening, midnight)
values (1, '2023-03-18', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (2, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (9155336, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3836513, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (18610653, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (14138382, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (28884781, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (8255293, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (4261247, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}');

-- auto-generated definition
create table if not exists ipm_complete_rl_predictions
(
    account_id  integer not null
        constraint ipm_complete_rl_predictions_pkey
            primary key,
    loaded_date date,
    morning     integer[],
    afternoon   integer[],
    evening     integer[],
    midnight    integer[]
);
insert into ipm_complete_rl_predictions (account_id, loaded_date, morning, afternoon, evening, midnight)
values (1, '2023-03-18', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (2, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (9155336, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (3836513, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (18610653, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (14138382, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (28884781, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (8255293, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}'),
       (4261247, '2023-03-19', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}', '{0, 1, 2, 3, 4}');


CREATE OR REPLACE FUNCTION f_get_home_hero_banners(p_customer_id INTEGER, p_day_interval_column TEXT)
    RETURNS TABLE
            (
                customer_id         INTEGER,
                rule_based_banners  int[],
                complete_rl_banners int[],
                guided_rl_banners   int[]
            )
AS
$$
    # variable_conflict use_column
BEGIN
    RETURN QUERY EXECUTE FORMAT('
        WITH rb_banners AS (
            SELECT account_id,
                   %I AS rule_based_banners
            FROM ipm_rule_based_predictions
            WHERE account_id = $1
        ),
        rl_banners AS (
            SELECT account_id,
                   %I AS complete_rl_banners
            FROM ipm_complete_rl_predictions
            WHERE account_id = $1
        ),
        guided_rl_banners AS (
            SELECT account_id,
                   %I AS guided_rl_banners
            FROM ipm_guided_rl_predictions
            WHERE account_id = $1
        )
        SELECT COALESCE(rbb.account_id, rlb.account_id, grlb.account_id) AS customer_id,
               COALESCE(rbb.rule_based_banners, ''{}'') AS rule_based_banners,
               COALESCE(rlb.complete_rl_banners, ''{}'') AS complete_rl_banners,
               COALESCE(grlb.guided_rl_banners, ''{}'') AS guided_rl_banners
        FROM rb_banners rbb
        FULL JOIN rl_banners rlb ON 1 = 1
        FULL JOIN guided_rl_banners grlb ON 1 = 1;',
                                p_day_interval_column, p_day_interval_column, p_day_interval_column
        ) USING p_customer_id;
END;
$$ LANGUAGE plpgsql;