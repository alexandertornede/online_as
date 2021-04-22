import unittest
from approaches.online.step_function import StepFunction
import numpy as np

class TestStepFunction(unittest.TestCase):

    def test_step_function(self):
        x = [1,5,10]
        y = [2,10,20]
        function = StepFunction(x=np.asarray(x), y=np.asarray(y))

        self.assertEqual(function.get_value(1), 2)
        self.assertEqual(function.get_value(3), 2)
        self.assertEqual(function.get_value(5), 10)
        self.assertEqual(function.get_value(7), 10)
        self.assertEqual(function.get_value(10), 20)
        self.assertEqual(function.get_value(20), 20)


if __name__ == '__main__':
    unittest.main()