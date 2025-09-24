"""function to display overall search
performance by country
"""
import streamlit as st
from src.pages import DATE_START
from src.data.processors import CountryMetric


def app():
    """this method will display overall
    country search metric
    """

    st.subheader("Performances by Country")

    @st.cache
    def country_metric_obj():
        return CountryMetric(DATE_START)

    col1, col2 = st.columns(2)

    col1.plotly_chart(
        country_metric_obj().figure_search_type(),
        use_container_width=True
    )

    col2.plotly_chart(
        country_metric_obj().figure_query_language(),
        use_container_width=True
    )
