drop table if exists item_replenishment;

create table if not exists item_replenishment
(
    account_id  integer not null
        constraint item_replenishment_pkey
            primary key,
    country_code varchar(6),
    loading_date date,
    category_id_list     TEXT[]
);

-- Inserting data into the item_replenishment table

INSERT INTO item_replenishment (account_id, country_code, loading_date, category_id_list)
VALUES
    (23735891, 'ae', '2023-11-14', '{1035fc72-c657-4bf4-a2c5-968618ec8b89,1c8fb3a0-a7dd-4d52-9299-0ad2995b90be,429b86ed-9b1f-4ae0-a8a0-621d45052ac8,7ab2fa01-af6d-4dce-9413-968c9ad5251e,8ea5bf94-9b85-46a8-86f9-8e7603691638,e3582e76-4300-4d75-b7eb-f2022bb7f663,1e57b123-dd6c-4ef8-a553-3655ae8bf49e}'),
    (1098952, 'ae', '2023-11-14', '{1035fc72-c657-4bf4-a2c5-968618ec8b89,1c8fb3a0-a7dd-4d52-9299-0ad2995b90be,8ea5bf94-9b85-46a8-86f9-8e7603691638,429b86ed-9b1f-4ae0-a8a0-621d45052ac8,1e57b123-dd6c-4ef8-a553-3655ae8bf49e,e3582e76-4300-4d75-b7eb-f2022bb7f663,d79a6aa3-7697-49bc-9372-a889e35f68a1}'),
    (1234, 'ae', '2023-11-14', '{}');
