"""function to run every included app
"""
from datetime import datetime
import streamlit as st
from src.pages import DATE_START
from . import clusters, country_metric


def app():
    """
    include all module app and run sequentially
    """

    st.info(
        f"Performances between {DATE_START} "
        f"and {datetime.today().strftime('%Y-%m-%d')}"
    )

    country_metric.app()

    st.markdown("""---""")

    clusters.app()


__all__ = ["app"]
