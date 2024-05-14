import unittest
import pandas as pd
from depression import sum_fau

class SumFAUTests(unittest.TestCase):

    def test_sum_fau(self):
        # import raw output from OpenFace
        df = pd.read_csv("russell_test.csv")

        # get the sum of the FAUs
        summed_fau = sum_fau(df)

        # perform assertion to check whether we have 17 FAUs 
        # used to test the correctness of the for loop
        self.assertEqual(summed_fau.shape[0], 17, "There should be 17 FAUs")

        # perform assertion to check whether the value of each FAU sum is 0 or more
        # (i.e. there should not be negative values)
        for i in range(17):
            self.assertEqual(summed_fau[i] >= 0, True, "FAU sum not be a negative value")


if __name__ == "__main__":
    unittest.main()