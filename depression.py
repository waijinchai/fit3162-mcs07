import streamlit as st
import pandas as pd
import numpy as np
import os
from OpenFace.waijin_extract_fau import extract_fau
import tensorflow as tf 
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import AudioFileClip
import speechDepressionMain
import speechDepressionMain.cluster
import speechDepressionMain.cluster.main_audio

def save_uploaded_file(uploaded_file):
    SAVE_PATH = "./OpenFace/input_videos/"
    try:
        with open(os.path.join(SAVE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        print(e)
        return False

def convert_mp4_to_wav(mp4_file_path, wav_file_path):
    # Usage: convert_mp4_to_wav(mp4_file_path='input_video_new.mp4', wav_file_path='output.wav')
    # Load the video file
    video_clip = AudioFileClip(mp4_file_path)
    
    # Extract the audio from the video and save it as a WAV file
    video_clip.write_audiofile(wav_file_path, codec='pcm_s16le')  # codec for WAV format

def sum_fau(df: pd.DataFrame) -> np.array:
    df.columns = [c.strip() for c in df.columns]

    FAU = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]

    for au in FAU:
        cond1 = df[au + "_c"] == 1
        cond2 = df[au + "_r"] >= 0.5
        cond3 = df["confidence"] >= 0.98
        cond4 = df["success"] == 1
        df[au] = np.where(cond1 & cond2 & cond3 & cond4, 1, 0)

    au_df = df.iloc[:, -17:]
    x = np.array(au_df.sum())

    return x

def vector_matching(df: pd.DataFrame) -> np.ndarray:
    df.columns = [c.strip() for c in df.columns]

    df_processed_duration = pd.read_csv("./VectorMatching/processed_duration_avg.csv")

    # get the duration of the video
    duration = df["timestamp"].iloc[-1]

    # get the average FAU count based on the duration of the video
    fau_sum_vector = sum_fau(df)
    fau_sum_vector = np.ceil(fau_sum_vector / duration).reshape(1, -1)

    # get the vectors for each category
    anxiety = np.array(df_processed_duration[df_processed_duration["Category"] == "Anxiety"].iloc[:, -17:])
    mild = np.array(df_processed_duration[df_processed_duration["Category"] == "Mild"].iloc[:, -17:])
    moderate = np.array(df_processed_duration[df_processed_duration["Category"] == "Moderate"].iloc[:, -17:])
    severe = np.array(df_processed_duration[df_processed_duration["Category"] == "Severe"].iloc[:, -17:])

    # compute the cosine similarity for each category
    anxiety_vector = cosine_similarity(anxiety, fau_sum_vector)
    mild_vector = cosine_similarity(mild, fau_sum_vector)
    moderate_vector = cosine_similarity(moderate, fau_sum_vector)
    severe_vector = cosine_similarity(severe, fau_sum_vector)

    # compile the cosine similarity vectors
    vectors = np.array([anxiety_vector.mean(), mild_vector.mean(), moderate_vector.mean(), severe_vector.mean()])
    
    return vectors

def predict_video(x: np.array) -> np.ndarray:
    x = x.reshape(1, -1)

    # normalise the input before feeding it to the model
    x_norm = preprocessing.normalize(x)
    print("Normalised input:\n", x_norm)

    # retrieve the deep learning model
    print("Loading model...")
    model_path = "./100_epoch_mlp.h5"
    model = tf.keras.models.load_model(model_path)
    print("Model loaded!")
    print("Starting prediction...")

    result = model.predict(x_norm)
    print("Predicted result:", result)

    return result

def get_category(array: np.ndarray) -> str:
    array_list = array.flatten().tolist()
    index = array_list.index(max(array_list))

    try:
        if index == 0:
            return "Anxiety"
        elif index == 1:
            return "Mild Depression"
        elif index == 2:
            return "Moderate Depression"
        elif index == 3:
            return "Severe Depression"
    except Exception as e:
        print(e)
        return "Invalid Index"
    
def map_fau_name(fau_array: np.array) -> pd.DataFrame:
    FAU = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]

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
    au_sums_df = pd.DataFrame({'AU_code': FAU, 'AU_sum': fau_array})
    au_sums_df['AU_name'] = au_sums_df['AU_code'].map(au_names)
    
    # Rearrange columns to have AU_name after AU_code
    au_sums_df = au_sums_df[['AU_code', 'AU_name', 'AU_sum']].set_index("AU_code")

    return au_sums_df


if __name__ == "__main__":
    st.title("AI in Depression and Anxiety Understanding")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        uploaded_file = st.file_uploader("Choose a .mp4 file", type=["mp4"])

        if uploaded_file is not None:
            # save the uploaded file in the OpenFace directory
            save_uploaded_file(uploaded_file)

            # convert the video into an audio file of type .wav and store it in the speech-main-depression directory
            video_filepath = f"./OpenFace/input_videos/{uploaded_file.name}"
            audio_filepath = f"./speechDepressionMain/test_data/{uploaded_file.name[:-4]}.wav"
            convert_mp4_to_wav(video_filepath, audio_filepath)

            # display the input video
            st.subheader(f"Video: {uploaded_file.name[:-4]}")
            st.video(uploaded_file)

            # use OpenFace to extract the FAUs from the input video
            extract_fau_state = st.text("Extracting FAUs...")
            if f"{uploaded_file.name[-4]}.csv" not in os.listdir("./OpenFace/output"):
                df = extract_fau(uploaded_file)
            else:
                df = pd.read_csv(f"./OpenFace/output/{uploaded_file.name[:-4]}.csv")
            df_fau_sum = sum_fau(df)
            extract_fau_state.text("Extracting FAUs...Done!!")

            st.dataframe(df)

    with col2:
        if uploaded_file is not None:
            st.write("")
            st.subheader("Statistics (Facial Action Units Count)")
            df_fau_mapped = map_fau_name(df_fau_sum)
            st.write(df_fau_mapped)  # Display the DataFrame with Action Unit names and sums
            st.subheader("Statistics Plot (Facial Action Units Count) ")
            st.line_chart(df_fau_mapped[['AU_sum']], y="AU_sum")
            st.subheader(f"Results (FAU Vector Matching)")
            vector_matching_result = vector_matching(df)
            vector_matching_result_list = vector_matching_result.flatten().tolist()
            vector_matching_prob = np.ceil(max(vector_matching_result_list) * 100)
            vector_matching_category = get_category(vector_matching_result)
            st.write(f"{vector_matching_category} | Probability: {vector_matching_prob}%")  # perform vector matching to predict the category
            st.subheader(f"Results (FAU Classifier)")
            video_model_result = predict_video(df_fau_sum)
            video_model_result_list = video_model_result.flatten().tolist()
            video_model_prob = np.ceil(max(video_model_result_list) * 100)
            video_mode_category = get_category(video_model_result)
            print(f"{video_mode_category} | Probability: {video_model_prob}%")
            st.write(f"{video_mode_category} | Probability: {video_model_prob}%")  # use tensorflow to predict the category and write out the results
            st.subheader(f"Results (Audio Classifier)")
            result, prob = speechDepressionMain.cluster.main_audio.predict_audio(f"{uploaded_file.name[:-4]}.wav")
            st.write(f"Depression/Anxiety Detected | Probability: {np.ceil(prob[0] * 100)}%" if result[0] == 1 else f"No Depression/Anxiety Detected | Probability: {np.ceil(prob[0] * 100)}%")

        else:  # default state when no inputs are uploaded yet
            st.subheader("Statistics (FAU)")
            st.subheader("Statistics Plot (FAU)")
            st.subheader(f"Results (FAU Vector Matching)")
            st.subheader(f"Results (FAU Classifier)")
            st.subheader(f"Results (Audio Classifier)")