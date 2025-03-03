import streamlit as st
import pandas as pd
from util import chat_gpt

with open("tab_icon.png", "rb") as image_file:
    icon_bytes = image_file.read()
st.set_page_config(
    page_title="RegressifyXpert",
    page_icon= icon_bytes,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://regressifyxpert.github.io/test/index.html',
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.header("Welcome to RegressifyXpert")
st.write("""
    We're dedicated to empowering your data-driven decisions through advanced regression analysis. Whether you're a seasoned analyst or just beginning your journey into data science, RegressifyXpert is here to support you every step of the way.
    """)

# Always show the image
# st.image("analysis.jpg", use_column_width=True)
col1,col2 = st.columns([0.7,0.3])
col1.subheader("How to do multiple regression analysis ?")
col2.link_button("Learn More", "https://regressifyxpert.github.io/test/EXAMPLE.html")
video_file = open('demo.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
    


chat_gpt()



pages = st.container(border=False )
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col5:
        if st.button("next page ▶️"):
            st.switch_page("pages/1_1️⃣_Data_Preprocessing.py")

