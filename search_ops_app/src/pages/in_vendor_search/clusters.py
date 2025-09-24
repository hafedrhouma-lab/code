"""function to automatically compute groups of best/worst
queries based on user's filters
"""
import os
import streamlit as st
import numpy as np
from src.data.processors import InVendorSearchQuery
from src.utils.helper import (
    create_gb_dataframe,
    figure_histogram,
    figure_cum_sum,
    apply_style_to_str,
)
from src.pages import GeographyMapping, UserSpecifications, DATE_START
from src.model.clustering import QuantileModel

__here__ = os.path.dirname(os.path.abspath(__file__))

GEOGRAPHY_MAPPING_FILE = os.path.join(
    __here__,
    "..",
    "resources/in_vendor_search_location_stores_mapping.csv"
)

geograpy_mapping_obj = GeographyMapping(GEOGRAPHY_MAPPING_FILE)


def map_selected_value(value: str, map_values: list) -> list:
    """returns values to query if value="All"""
    if value == "All":
        return map_values
    return [value]


def app():
    """this method will display groups of queries
    based on user's filters
    """

    # ----------- Query the data ---------------
    st.subheader("Queries Clustering")

    col1, col2, col3 = st.columns(3)
    os = col1.selectbox("OS", ["All"] + UserSpecifications().os_values)
    language = col2.selectbox(
        "Language", UserSpecifications().language_values + ["All"]
    )
    store_type = col3.selectbox(
        "Store Type", UserSpecifications().store_type_values + ["All"]
    )

    col1, col2, col3, col4 = st.columns(4)

    country = col1.selectbox("Country", geograpy_mapping_obj.get_countries())

    city = col2.selectbox("City", ["All"] + geograpy_mapping_obj.get_cities(country))

    area = col3.selectbox("Area", ["All"] + geograpy_mapping_obj.get_areas(country, city))

    store_name = col4.selectbox("Store name", ["All"] + geograpy_mapping_obj.get_stores(country))

    meta = {}

    if area != "All":
        meta["area"] = area

    if city != "All":
        meta["city"] = city

    if store_name != "All":
        meta["store_name"] = store_name

    @st.cache
    def search_query_obj():
        return InVendorSearchQuery(
            DATE_START,
            map_selected_value(os, getattr(UserSpecifications(), "os_values")),
            map_selected_value(language, getattr(UserSpecifications(), "language_values")),
            map_selected_value(
                store_type, getattr(UserSpecifications(), "store_type_values")
            ),
            country,
            meta,
        )


    quantile_clustering = QuantileModel(
        ["%cum_session", "ATC_percentage", "CVR_percentage", "ZRR_percentage"]
    )

    data_frame_clustered = quantile_clustering.predict(
        search_query_obj().get_most_searched()
    )

    # ----------- Metric Display ---------------
    st.markdown("Distributions of **key metrics**")
    st.info(
        f"All query search: {search_query_obj().number_search}. "
        f"**{search_query_obj().number_filtered_search}** "
        f"last query searches are skipped "
        "after keeping top 80%"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.plotly_chart(
        figure_cum_sum(search_query_obj().get_most_searched(), "%cum_session"),
        use_container_width=True,
    )
    col1.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: "
            f"{np.unique(data_frame_clustered['%cum_session_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    col2.plotly_chart(
        figure_histogram(search_query_obj().get_most_searched(), "ATC_percentage"),
        use_container_width=True,
    )
    col2.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: " f"{np.unique(data_frame_clustered['ATC_percentage_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    col3.plotly_chart(
        figure_histogram(search_query_obj().get_most_searched(), "CVR_percentage"),
        use_container_width=True,
    )
    col3.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: " f"{np.unique(data_frame_clustered['CVR_percentage_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    col4.plotly_chart(
        figure_histogram(search_query_obj().get_most_searched(), "ZRR_percentage"),
        use_container_width=True,
    )
    col4.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: "
            f"{np.unique(data_frame_clustered['ZRR_percentage_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("All **Data** Table")
    create_gb_dataframe(search_query_obj().get_most_searched(), "blue")

    # -----------Cluster selection ---------------
    st.markdown("**Cluster** display")
    col1, col2, col3 = st.columns(3)

    cum_session_quantile = col1.selectbox(
        "Session cluster",
        quantile_clustering.labels.get("%cum_session_quantile") + ["All"],
    )

    atc_quantile = col2.selectbox(
        "ATC cluster", quantile_clustering.labels.get("ATC_percentage_quantile") + ["All"]
    )

    cvr_quantile = col3.selectbox(
        "CVR cluster", quantile_clustering.labels.get("CVR_percentage_quantile") + ["All"]
    )

    create_gb_dataframe(
        data_frame_clustered[
            (
                    (
                        data_frame_clustered["%cum_session_quantile"].isin(
                            map_selected_value(
                                cum_session_quantile,
                                quantile_clustering.labels.get("%cum_session_quantile"),
                            )
                        )
                    )
                    & (
                        data_frame_clustered["ATC_percentage_quantile"].isin(
                            map_selected_value(
                                atc_quantile,
                                quantile_clustering.labels.get("ATC_percentage_quantile"),
                            )
                        )
                    )
                    & (
                        data_frame_clustered["CVR_percentage_quantile"].isin(
                            map_selected_value(
                                cvr_quantile,
                                quantile_clustering.labels.get("CVR_percentage_quantile"),
                            )
                        )
                    )
            )
        ].drop(
            columns=[
                "%cum_session_quantile",
                "ATC_percentage_quantile",
                "CVR_percentage_quantile",
                "ZRR_percentage_quantile"
            ]
        ),
        "streamlit",
    )

