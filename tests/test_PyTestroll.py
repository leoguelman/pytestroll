from pytestroll.pytestroll import test_size_nn
import numpy as np


def test_test_size_nn():
    """ Test test_size_nn"""


    expected = np.array([2283.89002819, 2283.89002819])
    actual = test_size_nn(N = 100000, s= np.sqrt(0.68*(1-0.68)), mu=0.68, sigma=0.03)
    
    assert(np.allclose(actual, expected, rtol=1e-05, atol=1e-08))
  

