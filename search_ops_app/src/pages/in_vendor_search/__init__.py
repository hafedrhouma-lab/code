"""function to run every included app
"""
from datetime import datetime
import streamlit as st
from src.pages import DATE_START
from . import clusters


def app():
    """
    include all module app and run sequentially
    """

    st.info(
        f"Performances between {DATE_START} "
        f"and {datetime.today().strftime('%Y-%m-%d')}"
    )

    clusters.app()


__all__ = ["app"]
