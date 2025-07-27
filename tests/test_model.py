import unittest
from app.model import predict

class TestIrisModel(unittest.TestCase):
    
    def test_predict_setosa(self):
        # Input that should predict class 0 (Setosa)
        input_data = [[5.1, 3.5, 1.4, 0.2]]
        result = predict(input_data)
        self.assertEqual(result, [0])

    def test_predict_versicolor(self):
        # Input that should predict class 1 (Versicolor)
        input_data = [[6.0, 2.2, 4.0, 1.0]]
        result = predict(input_data)
        self.assertEqual(result, [1])

    def test_predict_virginica(self):
        # Input that should predict class 2 (Virginica)
        input_data = [[6.9, 3.1, 5.4, 2.1]]
        result = predict(input_data)
        self.assertEqual(result, [2])

    def test_invalid_input_shape(self):
        # Input with wrong shape should raise an error
        input_data = [[1.0, 2.0]]
        with self.assertRaises(Exception):
            predict(input_data)

if __name__ == '__main__':
    unittest.main()
