import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
from OpenFace.waijin_extract_fau import extract_fau
import tensorflow as tf 
from sklearn import preprocessing

def save_uploaded_file(uploaded_file):
    SAVE_PATH = "./OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        print(e)
        return False


def sum_fau(df):
    df.columns = [c.strip() for c in df.columns]

    FAU = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]

    for au in FAU:
        cond1 = df[au + "_c"] == 1
        cond2 = df[au + "_r"] >= 0.5
        cond3 = df["confidence"] >= 0.98
        cond4 = df["success"] == 1
        df[au] = np.where(cond1 & cond2 & cond3 & cond4, 1, 0)

    au_df = df.iloc[:, -17:]
    x = np.array(au_df.sum()).reshape(1, -1)

    # Mapping Action Unit codes to their names to display on UI
    au_names = {
        'AU01': 'Inner Brow Raiser',
        'AU02': 'Outer Brow Raiser',
        'AU04': 'Brow Lowerer',
        'AU05': 'Upper Lid Raiser',
        'AU06': 'Cheek Raiser',
        'AU07': 'Lid Tightener',
        'AU09': 'Nose Wrinkler',
        'AU10': 'Upper Lip Raiser',
        'AU12': 'Lip Corner Puller',
        'AU14': 'Dimpler',
        'AU15': 'Lip Corner Depressor',
        'AU17': 'Chin Raiser',
        'AU20': 'Lip stretcher',
        'AU23': 'Lip Tightener',
        'AU25': 'Lips part',
        'AU26': 'Jaw Drop',
        'AU45': 'Blink',
    }

    # Create DataFrame with sums and add Action Unit names
    au_sums_df = pd.DataFrame({'AU_code': au_df.columns, 'AU_sum': x.flatten()})
    au_sums_df['AU_name'] = au_sums_df['AU_code'].map(au_names)
    
    # Rearrange columns to have AU_name after AU_code
    au_sums_df = au_sums_df[['AU_code', 'AU_name', 'AU_sum']].set_index("AU_code")

    return au_sums_df

def predict_video(x):
    # TODO: remember to normalise
    model_path = "./100_epoch_mlp.h5"
    model = tf.keras.models.load_model(model_path)
    return model.predict(x)

def get_category(array):
    array_list = array.flatten().tolist()
    index = array_list.index(1.0)

    if index == 0:
        return "Anxiety"
    elif index == 1:
        return "Mild Depression"
    elif index == 2:
        return "Moderate Depression"
    elif index == 3:
        return "Severe Depression"

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
            df_fau_sum = sum_fau(df)
            extract_fau_state.text("Extracting FAUs...Done!!")

            st.dataframe(df)

    with col2:
        if uploaded_file is not None:
            st.write("")
            st.subheader("Statistics (Facial Action Units Count)")
            st.write(df_fau_sum)  # Display the DataFrame with Action Unit names and sums
            st.subheader("Statistics Plot (Facial Action Units Count) ")
            st.line_chart(df_fau_sum[['AU_sum']], y="AU_sum")
            st.subheader(f"Results (FAU Vector Matching)")
            st.subheader(f"Results (FAU Classifier) ")
            # Use the sum of FAUs (x) to predict the category, as before 
            x = df_fau_sum[['AU_sum']].values.reshape(1, -1)
            st.write(get_category(predict_video(x)))  # use tensorflow to predict the category and write out the results
            st.subheader(f"Results (Audio Classifier) ")

        else:  # default state when no inputs are uploaded yet
            st.subheader("Statistics (FAU)")
            st.subheader("Statistics Plot (FAU) ")
            st.subheader(f"Results (FAU Vector Matching) ")
            st.subheader(f"Results (FAU Classifier) ")
            st.subheader(f"Results (Audio Classifier) ")