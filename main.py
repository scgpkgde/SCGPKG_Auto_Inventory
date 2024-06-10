import streamlit as st
from kraft_system import main as kraft_system_app
from duplex_system import main as duplex_system_app

PAGES = {
    "Kraft system": kraft_system_app,
    "Duplex system": duplex_system_app,
}
st.sidebar.write(
        f'<div style="margin-top: -35px;font: initial;font-size: 40px;color: #162e74;">PP Inventory',
        f'</div>',
        unsafe_allow_html=True
    )
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)
st.sidebar.write(
        f'<h2 style="color: #162e74;font: initial;font-size: 20px; margin-top: -10px">{selection}',
        f'</h2>',
        unsafe_allow_html=True
    )
page = PAGES[selection]
page()