import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
from OpenFace.waijin_extract_fau import extract_fau
import tensorflow as tf 

def save_uploaded_file(uploaded_file):
    SAVE_PATH = "./OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        print(e)
        return False


def merge_script(df):
    df.columns = [c.strip() for c in df.columns]

    FAU = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]


    for au in FAU:
        cond1 = df[au + "_c"] == 1
        cond2 = df[au + "_r"] >= 0.5
        cond3 = df["confidence"] >= 0.98
        cond4 = df["success"] == 1
        df[au] = np.where(cond1 & cond2 & cond3 & cond4, 1, 0)

    au_df = df.iloc[:, -17:]
    x = np.arrau(au_df.sum())
    return x


def predict(x):
    model_path = "saved_model.pb"
    model = tf.keras.models.load(model_path)
    return model.predict(x)

if __name__ == "__main__":
    st.title("AI in Depression and Anxiety Understanding")


    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        uploaded_file = st.file_uploader("Choose a .mp4 file", type=["mp4"])

        if uploaded_file is not None:
            save_uploaded_file(uploaded_file)

            st.subheader(f"Video: {uploaded_file.name[:-4]}")
            st.video(uploaded_file)

            extract_fau_state = st.text("Extracting FAUs...")
            df = extract_fau(uploaded_file)
            extract_fau_state.text("Extracting FAUs...Done!!")

            st.dataframe(df)

    with col2:

        if uploaded_file is not None:
            FAU_count = pd.DataFrame(df.iloc[:, 22:].sum(axis=0))
            FAU_count.columns = ["Count"]
            st.write ("")
            st.subheader("Statistics (Facial Action Units Count)")
            st.write(FAU_count)
            st.subheader("Statistics Plot ")
            st.line_chart(FAU_count, y="Count")
            st.subheader(f"Results (Vector Matching): ")
            st.markdown(
                """
                <style>
                .st-ef {
                    padding-bottom: 250px;
                }
                </style>
                """,
                
                unsafe_allow_html=True
            )
            st.subheader(f"Results (Classifier): ")
            
        else: 
            st.subheader("Statistics (Facial Action Units Count)")
            st.subheader("Statistics Plot ")
            st.subheader(f"Results (Vector Matching) ")
            st.subheader(f"Results (Classifier) ")