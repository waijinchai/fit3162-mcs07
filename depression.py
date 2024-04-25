import streamlit as st
import numpy as np
import pandas as pd
import os
import wget
from OpenFace.waijin_extract_fau import extract_fau

st.title("AI in Depression and Anxiety Understanding")

uploaded_file = st.file_uploader("Choose a file", type=["mp4"])


def save_uploaded_file(uploaded_file):
    SAVE_PATH = "C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        print(e)
        return False
  
# if uploaded_file is not None:
#     result = save_uploaded_file(uploaded_file)
#     if result:
#         st.success("File saved suffesscully!")
#     else:
#         st.error("Failed to save file!")

if uploaded_file is not None:
    save_uploaded_file(uploaded_file)

    st.subheader(f"Video: {uploaded_file.name[:-4]}")
    st.video(uploaded_file)

    extract_fau_state = st.text("Extracting FAUs...")
    df = extract_fau(uploaded_file)
    extract_fau_state.text("Extracting FAUs...Done!!")

    st.dataframe(df)