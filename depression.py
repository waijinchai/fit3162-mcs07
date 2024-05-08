import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
from OpenFace.waijin_extract_fau import extract_fau

def save_uploaded_file(uploaded_file):
    SAVE_PATH = "./OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    st.title("AI in Depression and Anxiety Understanding")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["mp4"])

        if uploaded_file is not None:
            save_uploaded_file(uploaded_file)

            st.subheader(f"Video: {uploaded_file.name[:-4]}")
            st.video(uploaded_file)

            extract_fau_state = st.text("Extracting FAUs...")
            df = extract_fau(uploaded_file)
            extract_fau_state.text("Extracting FAUs...Done!!")

            st.dataframe(df)

    with col2:
        st.subheader("Statistics")

        if uploaded_file is not None:
            FAU_count = pd.DataFrame(df.iloc[:, 22:].sum(axis=0))
            FAU_count.columns = ["Count"]

            st.write(FAU_count)
            st.line_chart(FAU_count, y="Count")