"""function to automatically compute groups of best/worst
queries based on user's filters
"""
import os
import streamlit as st
import numpy as np
from src.data.processors import SearchQuery
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
    "resources/country_city_area_mapping.csv"
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
    search_type = col3.selectbox(
        "Search Type", UserSpecifications().search_type_values + ["All"]
    )

    col1, col2, col3 = st.columns(3)

    country = col1.selectbox("Country", geograpy_mapping_obj.get_countries())

    city = col2.selectbox("City", ["All"] + geograpy_mapping_obj.get_cities(country))

    area = col3.selectbox("Area", ["All"] + geograpy_mapping_obj.get_areas(country, city))

    meta = {}

    if area != "All":
        meta["area"] = area

    if city != "All":
        meta["city"] = city

    search_query_obj = SearchQuery(
        DATE_START,
        map_selected_value(os, getattr(UserSpecifications(), "os_values")),
        map_selected_value(language, getattr(UserSpecifications(), "language_values")),
        map_selected_value(
            search_type, getattr(UserSpecifications(), "search_type_values")
        ),
        country,
        meta,
    )

    quantile_clustering = QuantileModel(
        ["%cum_session", "CTR(%)", "CVR2(%)", "session_CVR(%)"]
    )

    data_frame_clustered = quantile_clustering.predict(
        search_query_obj.get_most_searched()
    )

    # ----------- Metric Display ---------------
    st.markdown("Distributions of **key metrics**")
    st.info(
        f"All query search: {search_query_obj.number_search}. "
        f"**{search_query_obj.number_filtered_search}** "
        f"last query searches are skipped "
        "after keeping top 80%"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.plotly_chart(
        figure_cum_sum(search_query_obj.get_most_searched(), "%cum_session"),
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
        figure_histogram(search_query_obj.get_most_searched(), "CTR(%)"),
        use_container_width=True,
    )
    col2.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: " f"{np.unique(data_frame_clustered['CTR(%)_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    col3.plotly_chart(
        figure_histogram(search_query_obj.get_most_searched(), "CVR2(%)"),
        use_container_width=True,
    )
    col3.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: " f"{np.unique(data_frame_clustered['CVR2(%)_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    col4.plotly_chart(
        figure_histogram(search_query_obj.get_most_searched(), "session_CVR(%)"),
        use_container_width=True,
    )
    col4.markdown(
        apply_style_to_str(
            "small-font",
            f"Quantiles: "
            f"{np.unique(data_frame_clustered['session_CVR(%)_quantile'])}",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("All **Data** Table")
    create_gb_dataframe(search_query_obj.get_most_searched(), "blue")

    # -----------Cluster selection ---------------
    st.markdown("**Cluster** display")
    col1, col2, col3, col4 = st.columns(4)

    cum_session_quantile = col1.selectbox(
        "Session cluster",
        quantile_clustering.labels.get("%cum_session_quantile") + ["All"],
    )

    ctr_quantile = col2.selectbox(
        "CTR cluster", ["All"] + quantile_clustering.labels.get("CTR(%)_quantile")
    )

    cvr2_quantile = col3.selectbox(
        "CVR2 cluster", quantile_clustering.labels.get("CVR2(%)_quantile") + ["All"]
    )

    session_cvr_quantile = col4.selectbox(
        "Session CVR cluster",
        quantile_clustering.labels.get("session_CVR(%)_quantile") + ["All"],
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
                        data_frame_clustered["CTR(%)_quantile"].isin(
                            map_selected_value(
                                ctr_quantile,
                                quantile_clustering.labels.get("CTR(%)_quantile"),
                            )
                        )
                    )
                    & (
                        data_frame_clustered["CVR2(%)_quantile"].isin(
                            map_selected_value(
                                cvr2_quantile,
                                quantile_clustering.labels.get("CVR2(%)_quantile"),
                            )
                        )
                    )
                    & (
                        data_frame_clustered["session_CVR(%)_quantile"].isin(
                            map_selected_value(
                                session_cvr_quantile,
                                quantile_clustering.labels.get("session_CVR(%)_quantile"),
                            )
                        )
                    )
            )
        ].drop(
            columns=[
                "%cum_session_quantile",
                "CTR(%)_quantile",
                "CVR2(%)_quantile",
                "session_CVR(%)_quantile",
            ]
        ),
        "streamlit",
    )
