import subprocess
import os
import pandas as pd
"""
Reference - https: //github.com/TadasBaltrusaitis/OpenFace 

Uses command line to run Tadas OpenFace FeatureExtraction.exe
Get original output, Get selected column output for FAUs in output_csv dir

Running this file: 
    cd to claire_test_commands
    modify input_video_dir and output_dir 
    run this script

"""

def ensure_dir(directory):
    """
    Ensure the output directory exists. If not, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_csv(input_file_path, columns_to_select):
    """
    Creates a new CSV file containing only specified columns from the original output CSV.
    It dynamically checks for the existence of columns and selects only those present.

    Parameters:
    - input_file_path: Path to the original CSV file
    - columns_to_select: List of columns to include in the new CSV
    """
    # Read the original FeatureExtraction.exe output CSV file
    data = pd.read_csv(input_file_path)

    # Select only the columns that exist in the CSV, 
    existing_columns = [col for col in columns_to_select if col in data.columns]
    selected_data = data[existing_columns]

    # Report any missing columns
    missing_columns = set(columns_to_select) - set(existing_columns)
    if missing_columns:
        print(f"Warning: The following columns were not found in the file and have been skipped: {missing_columns}")

    # Construct the new filename and save the selected data
    new_file_path = input_file_path.replace('.csv', '_selected_columns.csv')
    selected_data.to_csv(new_file_path, index=False)
    print(f"Selected columns CSV saved to {new_file_path}")


def extract_features(video_folder, output_dir):
    """
    Extracts features from video files using OpenFace and saves the output in individual directories named after each video file.

    Parameters:
    - video_folder: Directory containing the video files to be processed
    - output_dir: Base directory where the output folders will be created
    """
    feature_extraction_executable = "C:\\OpenFace_2.2.0_win_x64\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe"

    # Ensure the base output directory exists
    ensure_dir(output_dir)
    
    # Define columns to select for the new CSV
    basic_columns = ['frame', ' face_id', ' timestamp', ' confidence', ' success']
    intensity_columns = [f' AU0{num}_r' if num <10 else f' AU{num}_r' for num in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)]
    presence_columns = [f' AU0{num}_c' if num <10 else f' AU{num}_c'for num in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45)]
    columns_to_select = basic_columns + intensity_columns + presence_columns

    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.avi', '.mp4', '.mov'))]

    for video_file in video_files:

        video_name = os.path.splitext(os.path.basename(video_file))[0]
        this_output_folder = os.path.join(output_dir, video_name)  # NOTE: Each input video has its corresponding output folder
        ensure_dir(this_output_folder)

        command = [feature_extraction_executable, "-f", video_file, "-out_dir", this_output_folder]
        
        # Executing the command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print(f"Feature extraction for {video_name} completed successfully.")
            # Process the original output CSV file 
            csv_files = [f for f in os.listdir(this_output_folder) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(this_output_folder, csv_file)
                print (csv_path)
                process_csv(csv_path, columns_to_select)
        else:
            print(f"Feature extraction for {video_name} failed: {result.stderr}")

# NOTE: Set path to input and output folder on your machine
input_video_dir = "C:\\FYP2Group\\fit3162-mcs07\\OpenFace_2.2.0_win_x64\\claire_testing_commandline\\input_videos" # copy path of "input_videos" dir
output_dir = "C:\\FYP2Group\\fit3162-mcs07\\OpenFace_2.2.0_win_x64\\claire_testing_commandline\\output_csv" # copy path of "output_csv" dir


# Run the featureExtraction.exe and process the output
extract_features(input_video_dir, output_dir)
