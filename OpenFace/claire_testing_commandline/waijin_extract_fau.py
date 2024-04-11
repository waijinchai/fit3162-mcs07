# import required libraries
import subprocess
import pandas as pd

# set the directories
input_dir = "C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/claire_testing_commandline/input_videos"   # directory of where the videos are
output_dir = "C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/claire_testing_commandline/output"  # directory to output the csv file
of_dir = "C:/Users/User/Desktop/MonashDocuments/FIT3162/Code/OpenFace_2.2.0_win_x64/FeatureExtraction.exe"  # directory of where the FeatureExtraction.exe is at

# run the script using the command line
cmd_str = f"""for /F %i in ('dir /b "{input_dir}"') do "{of_dir}" -inroot "{input_dir}" -f %i  -aus -out_dir "{output_dir}" """
subprocess.run(cmd_str, shell=True, text=True, stdout=True)

# import the output csv
df = pd.read_csv("C:/Users/User/Desktop/MonashDocuments/FIT3162/Repo/fit3162-mcs07/OpenFace/claire_testing_commandline/output/russell_test.csv")
# print(df.sample(n=5))