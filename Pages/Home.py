import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title=" VizAI by Ethicallogix - Home",
    page_icon="📊",
    layout="wide"
)

# ------------------- User Session Check -------------------
if not st.session_state.get('user'):
    st.switch_page("main")
else:
    user = st.session_state.user

    # ------------------- Page Styling -------------------
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=0.8">
        <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 3rem;
            }
            .main {
                background-color: #f4f9ff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ------------------- Page Header -------------------
    st.title("🏠 Welcome to Data App")
    st.divider()

    # Add vertical spacing for better layout
    st.markdown("## ")
    st.markdown("## ")

    # ------------------- Centered Box -------------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("## 📊 Data Analysis Dashboard")
            st.write(
                "VizAI by Ethicallogix is an intelligent data visualization platform powered by advanced AI analytics., "
                "It helps users transform raw datasets into meaningful, interactive dashboards — instantly.",

            )
            st.markdown("---")

            # Center the button horizontally
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("Get Started", use_container_width=True):
                    st.switch_page("Data Cleaning")  # ✅ Correct path for Streamlit multipage apps
