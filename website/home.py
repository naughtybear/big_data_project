import streamlit as st
from predict import render_predict_page
from explore import show_explore_page
import os

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
if page == "Predict":
    st.title("Movie Box Office Prediction")
    image_path = os.path.join("assets", "box_office.png")
    st.image(image_path, use_column_width=True)

    render_predict_page()
else:
    show_explore_page()