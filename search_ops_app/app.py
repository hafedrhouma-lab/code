import streamlit as st

# Custom imports
from src.app.multipage import MultiPage
from src.pages import homepage_search, in_vendor_search
from src.utils.helper import load_css

# Create an instance of the app
app = MultiPage()

# Main page
st.set_page_config(layout="wide")
load_css("src/css/style.css")

st.title("Grocery Search Ops")

# Add all your application here
app.add_page("In Vendor Search", in_vendor_search.app)
app.add_page("Homepage Search", homepage_search.app)

# The main app
app.run()
