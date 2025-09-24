"""contains helper functions"""
import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
import plotly.express as px
import plotly.graph_objects as go


def load_css(file_name):
    """load css file"""
    with open(file_name) as file:
        st.markdown(
            f'<style>{file.read()}</style>',
            unsafe_allow_html=True
        )


def apply_style_to_str(
        class_name: str, input_string:str):
    """apply css style on string"""
    return f"<p class='{class_name}'>{input_string}</p>"


def delete_session_state():
    """delete all session state variables"""
    for key in st.session_state.keys():
        del st.session_state[key]


def create_gb_dataframe(
        data_frame: pd.DataFrame,
        theme: str) -> AgGrid:
    """create st_grid dataframe"""
    g_b = GridOptionsBuilder.from_dataframe(data_frame)
    g_b.configure_pagination(enabled=True)
    g_b.configure_selection(use_checkbox=True)
    grid_options = g_b.build()

    return AgGrid(
        data_frame,
        gridOptions=grid_options,
        theme=theme,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True
    )


def figure_histogram(
        data_frame: pd.DataFrame,
        col: str) -> px:
    """distribution plot"""
    return px.histogram(
        data_frame,
        x=col,
        nbins=15,
        histnorm='percent',
        height=260,
        title=f"{col} distribution",
    ).update_traces(
        marker_line_width=1,
        marker_line_color="white"
    )


def figure_cum_sum(
        data_frame: pd.DataFrame,
        col: str) -> go:
    """cumsulative sum plot"""
    return go.Figure(
        data=[
            go.Histogram(
                x=data_frame[col],
                cumulative_enabled=True,
                nbinsx=10
            )
        ],
        layout=go.Layout(
            title=f"{col} cumulative sum",
            height=260
        )
    ).update_traces(
        marker_line_width=1,
        marker_line_color="white",
    )
