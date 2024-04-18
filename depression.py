import streamlit as st
import numpy as np
import pandas as pd

st.title("AI in Depression and Anxiety Understanding")

uploaded_file = st.file_uploader("Choose a file", type=["mp4"])

st.subheader(f"Video: {uploaded_file.name[:-4]}")
st.video(uploaded_file)

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