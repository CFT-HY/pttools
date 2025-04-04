"""Unit tests for assertions"""

import unittest

import numpy as np

from tests.utils.assertions import assert_allclose


class TestAllclose(unittest.TestCase):
    """Test the assert_allclose function"""
    def test_float(self):
        assert_allclose(1.1, 1.1)
        with self.assertRaises(AssertionError):
            assert_allclose(1.1, 1.2)

    # def test_list(self):
    #     actual = [1, 1.2, 1.3]
    #     desired = [1, 1.2, 1.4]
    #     with self.assertRaises(ValueError):
    #         assert_allclose(actual, desired)
    #
    # def test_nested_list(self):
    #     actual = [[1, 2], [3, 4]]
    #     desired = [[1, 2], [3, 5]]
    #     assert_allclose(actual, actual)
    #     with self.assertRaises(ValueError):
    #         assert_allclose(actual, desired)

    def test_ndarray_1d(self):
        actual = np.array([1, 1.1, 1.2])
        desired = np.array([1, 1.1, 1.3])
        assert_allclose(actual, actual)
        with self.assertRaises(AssertionError):
            assert_allclose(actual, desired)

    def test_ndarray_2d(self):
        actual = np.array([[1, 2], [3, 4]])
        desired = np.array([[1, 2], [3, 5]])
        assert_allclose(actual, actual)
        with self.assertRaises(AssertionError):
            assert_allclose(actual, desired)
