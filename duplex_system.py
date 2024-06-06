import streamlit as st
import streamlit_shadcn_ui as ui
from dashboard import data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6, data_table_7, data_table_8, data_table_9
import dashboard.data_preparation as data_preparation
import streamlit as st
from datetime import datetime, date, timedelta
import scipy.stats as stat
import pandas as pd

def main():
    st.title("Duplex Page")
    st.write("Welcome to duplex")        
    

