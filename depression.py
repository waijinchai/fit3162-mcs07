import os
import streamlit as st
import numpy as np
import pandas as pd
from OpenFace.waijin_extract_fau import extract_fau

st.title("AI in Depression and Anxiety Understanding")

uploaded_file = st.file_uploader("Choose a file", type=["mp4"])


def save_uploaded_file(uploaded_file):
    SAVE_PATH = "C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.get_buffer())
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


st.subheader(f"Video: {uploaded_file.name[:-4]}")
st.video(uploaded_file)

extract_fau_state = st.text("Extracting FAUs...")

df = extract_fau(uploaded_file)

extract_fau_state.text("Extracting FAUs...Done!!")


# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     df = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     df.rename(lowercase, axis="columns", inplace=True)
#     df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
#     return df

# # create a text element and let the user know the data is loading
# data_load_state = st.text("Loading data...")

# # Load 10,000 rows of data into the dataframe
# df = load_data(10000)

# # notify the user that the data was successfully loaded
# data_load_state.text("Done! (using st.cache_data)")

# if st.checkbox("Show raw data"):
#     st.subheader('Raw Data')
#     st.write(df)