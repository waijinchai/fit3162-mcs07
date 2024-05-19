import unittest
import os
from unittest.mock import patch
from depression import convert_mp4_to_wav

class AudioConversionTest(unittest.TestCase):

    def test_audio_conversion(self):
        video_file_path = 'OpenFace/input_videos/clairevid0.mp4'
        output_file_path = './OpenFace/output/clairevid0.wav'

        # converts the video file into an audio file
        convert_mp4_to_wav(video_file_path, output_file_path)

        # perform assertion to check whether file has been converted successfully
        self.assertIn("S06_1001-1.wav", os.listdir("./OpenFace/output"), "Audio file does not exists")

if __name__ == "__main__":
    unittest.main()