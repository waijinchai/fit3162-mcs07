import unittest
from depression import vector_matching
import pandas as pd

class VectorMatchingTest(unittest.TestCase):

    def test_vector_matching(self):
        # read in the extracted FAUs from a sample video from the dataset
        df = pd.read_csv("./OpenFace/output/S06_4002-06.csv")
        
        # compute the cosine similarity between he extracted FAUs and the different catagories
        result = vector_matching(df)

        # perform assertion to check whether the result is an array that contains 4 elements
        self.assertEqual(len(result), 4, "The result should only have 4 elements")

        # perform assertion to check whether the values of the elements in the result are >=0 and <=1
        for cos_sim in result:
            self.assertEqual((cos_sim >= 0 and cos_sim <= 1), True, "Cosine similarity values should be between 0 and 1 inclusive")


if __name__ == "__main__":
    unittest.main()