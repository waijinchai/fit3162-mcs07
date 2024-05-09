# import required libraries
import subprocess
import pandas as pd

# set the directories
input_dir = "./OpenFace/input_videos"   # directory of where the videos are
output_dir = "./OpenFace/output"  # directory to output the csv file
of_dir = "./OpenFace/OpenFace_2.2.0_win_x64/FeatureExtraction.exe"  # directory of where the FeatureExtraction.exe is at

def extract_fau(input_video) -> pd.DataFrame:
    print(input_video)
    cmd_str = f""" "{of_dir}" -f "{input_dir}/{input_video.name}" -out_dir "{output_dir}" -aus"""
    subprocess.run(cmd_str, shell=True, text=True, stdout=True)

    input_vid_name = input_video.name[:-4]
    df = pd.read_csv(f"./OpenFace/output/{input_vid_name}.csv")  

    return df  

if __name__ == "__main__":
    # # run the script using the command line
    cmd_str = f"""for /F %i in ('dir /b "{input_dir}"') do "{of_dir}" -inroot "{input_dir}" -f %i  -aus -out_dir "{output_dir}" """
    subprocess.run(cmd_str, shell=True, text=True, stdout=True)

    # import the output csv
    df = pd.read_csv("C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/output/clairevid0.csv")
    print(df.sample(n=5))
    # pass