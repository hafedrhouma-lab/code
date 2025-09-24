-- V1

create table vl_ranking_static_features
(
    vendor_id               integer,
    vendor_name             text,
    country_id              integer,
    city_id                 integer,
    area_id                 integer,
    is_tgo                  integer,
    rating_display          numeric,
    rating_count            integer,
    popularity_score        numeric,
    retention               numeric,
    timeofday_midnightsnack numeric,
    timeofday_breakfast     numeric,
    timeofday_lunch         numeric,
    timeofday_eveningsnack  numeric,
    timeofday_dinner        numeric,
    aov                     numeric,
    fail_rate               numeric,
    timezone_name           varchar(100)
);

create index vl_ranking_static_features_stg_country_id_idx
    on vl_ranking_static_features (country_id);

create index vl_ranking_static_features_stg_vendor_id_idx
    on vl_ranking_static_features (vendor_id);


insert into public.vl_ranking_static_features (vendor_id, vendor_name, country_id, city_id, area_id, is_tgo, rating_display, rating_count, popularity_score, retention, timeofday_midnightsnack, timeofday_breakfast, timeofday_lunch, timeofday_eveningsnack, timeofday_dinner, aov, fail_rate, timezone_name)
values  (15011, 'Abdoun Restaurant & Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 1.375, 1, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (34657, 'SMS Restaurant & Confectionery, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (38054, 'New Delma Restaurant & Cafeteria, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (2175, 'Tony Roma`s, Sheikh Zayed Road 1', 4, 35, 1280, 0, 3.6057692338258791, 52, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (20758, 'Moombai & Co, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (14197, 'Round Table Pizza, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 2.7083333333333335, 3, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (11535, 'Saj Bistro, Dubai World Trade Centre', 4, 35, 1280, 0, 3.68359375, 16, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (33849, 'Al Safeer Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 0.78125, 2, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (37105, 'Jashan, Dubai World Trade Center,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.0305645276181883, 13, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (35421, 'Jashan, Dubai World Trade Center', 4, 35, 1280, 0, 3.780555558445478, 33, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (35570, 'Fish & Co., Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 2.275, 5, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (37108, 'Fish & Co. , Dubai World Trade Center', 4, 35, 1280, 1, 3.9160879629629624, 24, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (643549, 'Ministry Of Burger, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (47302, 'Sakura Japanese Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (47300, 'Al fresco italian trattoria, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.9458333333333333, 8, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (2644, 'Wagamama, Sheikh Zayed Road', 4, 35, 1280, 1, 3.8794209725121087, 410, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (15229, 'Pizza Station - Dubai World Trade Center', 4, 35, 1280, 0, 3.4375, 9, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (11992, 'Jashan, Sheikh Zayed Road', 4, 35, 1280, 0, 2.4520833333333334, 6, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (34063, 'Headlines Bistro ِAnd Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (4665, 'Japengo Cafe, Dubai World Trade Centre', 4, 35, 1280, 0, 4.2152778333333334, 6, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (21495, 'Japengo Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.25, 11, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (22839, 'Frings, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.9166666666666665, 3, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (604229, 'F E L Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (21821, 'Keif Cafe,Dubai World Trade Center', 4, 35, 1280, 1, 4.1487179487179491, 13, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (24126, 'Indus Restaurant, Dubai World Trade Center', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (8919, 'Mazaher Restaurant, Dubai World Trade Centre', 4, 35, 1280, 0, 3.2638888888888888, 15, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (656646, 'LDC, Aspin, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.6624503968253972, 168, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (45492, 'The Noodle House, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (667944, 'Karakccino Prime Restaurant and Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.6, 5, null, 0, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (633298, 'Al Safadi, DIFC,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.21776649746193, 197, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (20783, 'Chili’s, DWTC Sheikh Zayed Road', 4, 35, 1280, 1, 3.9967948717948718, 13, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (602508, 'Il Caffe Di Roma, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (9249, 'Massaad BBQ Restaurant, Dubai World Trade Center', 4, 35, 1280, 1, 3.6954101536458333, 128, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (42130, 'Slider HQ, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (2392, 'Sasha''s Cafe, Sheikh Zayed Road 1', 4, 35, 1280, 0, 2.7056451612903225, 31, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (2728, 'Little Bangkok, Sheikh Zayed Road', 4, 35, 1280, 0, 3.55703552597738, 183, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (636914, 'Chef Liang Cantonese, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.166666666666667, 3, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (40770, 'Pablo De’ Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.2571428571428571, 35, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (673596, 'Caracas Patisserie N Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3, 2, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (643327, 'GOI, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 5, 4, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (635615, 'Wow Kebabs N Curries, Business Bay', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (645274, 'All The Perks Espresso Cafe, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (15158, 'Laval Lounge Restaurant - Sheikh Zayed Road', 4, 35, 1280, 0, 4.0773611111111112, 30, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (49717, 'O`DONER, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.3174603174603177, 21, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (28343, 'Mr. Brisket, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (615588, 'Masala Tiffin, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 2, 4, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (14825, 'Kiza Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 1.89375, 10, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (688836, 'Basha, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (693712, 'CHA CHA CHAI CAFE - DIFC, DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (693710, 'CHA CHA CHAI CAFE - SZR', 4, 35, 1280, 1, 1, 1, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (29752, 'Cafe Bateel,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.1833478019193375, 96, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (669020, 'The Burger Load by Kitch-In, DIFC', 4, 35, 1280, 1, 3.8125, 16, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (2978, 'Cafe Bateel, Sheikh Zayed Road 1', 4, 35, 1280, 0, 3.9775654269972445, 121, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (669016, 'Greek Street by Kitch-In,  DWTC', 4, 35, 1280, 1, 2, 6, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (673047, 'Sushi Sensei, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.387096774193548, 31, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (23393, 'Piadera FZE', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (679355, 'Ciao One Central, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (20375, 'Villa Beirut', 4, 35, 1280, 1, 3.3171027190925542, 43, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (679588, 'Motto,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (679631, 'More Cafe - Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 5, 10, null, null, null, null, null, null, null, null, 1, 'Asia/Dubai'),
        (640137, 'Beit El Kell Restaurant & Coffee Shop, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.5690476190476179, 14, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (678764, 'Royal Beans Coffee, Dubai World Trade Center', 4, 35, 1280, 1, 4.5, 2, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (679358, 'Street Burger, One Central, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (38061, 'Al Amoor Express, Sheikh Zayed, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 4.1069444444444434, 15, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (38741, 'Al Amoor Express, Sheikh Zayed ,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.713216485450316, 2273, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (634699, 'O''Doner, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 2.4333333333333331, 10, null, null, null, null, null, null, null, null, 1, 'Asia/Dubai'),
        (608232, 'Abou Afif Sandwiches From Beirut, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.699074074074074, 33, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (620215, 'BRONSON,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.2222222222222228, 6, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (611614, 'Buono Buono, Dubai World Trade Center', 4, 35, 1280, 1, 4.3253968253968251, 63, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (669937, 'Saad Enab Restaurant LLC,Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (31217, 'Burger King, Sheikh Zayed Road 1,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (8524, 'Pizza Hut, Sheikh Zayed Road 1', 4, 35, 1280, 0, 3.396955128205128, 26, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (1284, 'Which Wich, Sheikh Zayed Road', 4, 35, 1280, 0, 3.5775439698492462, 199, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (15124, 'Dao Xiang Restaurant, Dubai World Trade Centre', 4, 35, 1280, 0, 3.9887518658720875, 642, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (11407, 'Test Test Test Test', 4, 35, 1280, 0, 3.9791666666666665, 15, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (25142, 'COLDSTONE CREAMERY,Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (12989, 'The Pizza Company , Sheikh Zayed Road', 4, 35, 1280, 0, 2.9680681810957013, 55, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (43912, 'Zefki Levantine Eatery Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 4.064015151515151, 11, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (13143, 'Broccoli Pizza & Pasta, Sheikh Zayed Road', 4, 35, 1280, 0, 3.3343143458406637, 790, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (36995, 'Red Burger, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 2.9739583333333335, 8, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (459, 'Johnny Rockets, Sheikh Zayed Road', 4, 35, 1280, 0, 3.3863427124477168, 296, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (20890, 'Piadera, Dubai World Trade Centre-DWTC', 4, 35, 1280, 0, 3.55625, 5, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (38963, 'Margherita, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.2361111111111107, 36, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (600057, 'Moe''s On The 5th, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 2.3333333333333335, 1, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (601235, 'Nido, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (17392, 'Maw , Dubai World Trade Center - DWTC', 4, 35, 1280, 0, 3.74114056055493, 127, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (623741, 'Country Fried Chicken, World Trade Center', 4, 35, 1280, 1, 2.5, 8, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (47286, 'Abdoun Restaurant And Cafe,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.4663978494623651, 31, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (652726, 'Juice Bar, Sheiksh Zayed Road, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (663893, 'Daalchini, century village', 4, 35, 1280, 1, 3.6666666666666665, 6, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (636084, 'India Bistro, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 4.2222222222222223, 9, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (642015, 'Borsch Russian Restaurant, Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 0, 0, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (36623, 'Moombai & Co,Dubai World Trade Center - DWTC', 4, 35, 1280, 1, 3.0666666666666669, 5, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (611002, 'Uncle Deek, DWTC', 4, 35, 1280, 1, 3.850841750841751, 33, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (621135, 'Cafe Sushi', 4, 35, 1280, 1, 3.0833333333333335, 4, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (627553, 'C Restaurant, Al Awir', 4, 35, 1280, 1, 3.1166666666666667, 5, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (1165, 'PizzaExpress, Dubai World Trade Centre', 4, 35, 1280, 0, 3.7231326649686021, 299, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (638497, 'Mac & Wings, DWTC', 4, 35, 1280, 1, 4.25, 8, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (13021, 'Ayoush, Dubai World Trade Centre', 4, 35, 1280, 0, 4.1554350672228679, 902, null, null, null, null, null, null, null, null, null, 'Asia/Dubai'),
        (6581, 'Sushi Counter , Dubai World Trade Centre', 4, 35, 1280, 0, 3.7013372478799735, 73, null, null, null, null, null, null, null, null, null, 'Asia/Dubai');

create or replace function f_get_vendor_ranking_features(p_country_id integer, p_vendors integer[])
    returns TABLE
            (
                country_id       integer,
                vendor_id        integer,
                vendor_rank      integer,
                rating_display   numeric,
                fail_rate        numeric,
                popularity_score numeric,
                retention        numeric,
                rating_count     integer,
                aov_eur          numeric,
                is_tgo           integer,
                timeofday_score  numeric
            )
    language plpgsql
as
$$
    # variable_conflict use_column
begin
    return query
        select v.country_id,
               vendor_id,
               1                                    vendor_rank,
               rating_display,
               fail_rate,
               popularity_score,
               retention,
               rating_count,
               aov                                  aov_eur,
               is_tgo,
               CASE
                   when CAST(now() at time zone timezone_name as time) BETWEEN '00:00' AND '06:59'
                       then timeofday_midnightsnack
                   when CAST(now() at time zone timezone_name as time) BETWEEN '07:00' AND '11:59'
                       then timeofday_breakfast
                   when CAST(now() at time zone timezone_name as time) BETWEEN '12:00' AND '15:59'
                       then timeofday_lunch
                   when CAST(now() at time zone timezone_name as time) BETWEEN '16:00' AND '18:59'
                       then timeofday_eveningsnack
                   when CAST(now() at time zone timezone_name as time) BETWEEN '19:00' AND '23:59'
                       then timeofday_dinner END as timeofday_score
        from vl_ranking_static_features v
        where country_id = p_country_id
          and vendor_id = any (p_vendors);


end;
$$;