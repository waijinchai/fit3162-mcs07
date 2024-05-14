import unittest
import numpy as np
from depression import get_category

__author__ = "Chai Wai Jin"

class CategoryTests(unittest.TestCase):

    def test_anxiety(self):
        anxiety_result = np.array([[7.4092728e-01], [7.3483934e-11], [8.3307754e-03], [2.5074202e-01]])

        # perform assertion to check if the output category is correct
        self.assertEqual(get_category(anxiety_result), "Anxiety", "Should be Anxiety")
    
    def test_mild(self):
        mild_result = np.array([[1.4643296e-16], [9.9892539e-01], [1.0398891e-03], [3.4639601e-05]])

        # perform assertion to check if the output category is correct
        self.assertEqual(get_category(mild_result), "Mild Depression", "Should be Mild Depression")
    
    def test_moderate(self):
        moderate_result = np.array([[1.6080204e-01], [1.0497092e-04], [8.3547056e-01], [3.6224115e-03]])

        # perform assertion to check if the output category is correct
        self.assertEqual(get_category(moderate_result), "Moderate Depression", "Should be Moderate Depression")

    def test_severe(self):
        severe_result = np.array([[4.1010098e-11], [1.3806324e-11], [8.3816376e-06], [9.9999166e-01]])

        # perform assertion to check if the output catogery is correct
        self.assertEqual(get_category(severe_result), "Severe Depression", "Shouled be Severe Depression")

if __name__ == "__main__":
    unittest.main()

