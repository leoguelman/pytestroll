from pytestroll.pytestroll import (
    tr_size_nn, 
    profit_nn,
    nht_size_nn,
    profit_perfect_nn,
    error_rate_nn,
    profit_nn_sim
    )
from numpy.testing import assert_almost_equal
import numpy as np


def test_tr_size_nn():
    """ Test tr_size_nn"""
    
    y_ = np.array([2283.89002819, 2283.89002819])
    y = tr_size_nn(N = 100000, s=np.sqrt(0.68*(1-0.68)), mu=0.68, sigma=0.03)
    
    assert_almost_equal(y_, y)



def test_profit_nn():
    """ Test profit_nn"""
    
    y_ = np.float64(0.6153599861754739)
    y = profit_nn(n=np.array([2283.89002819, 2283.89002819]), N=100000, 
              s=np.sqrt(0.68*(1-0.68)), mu=0.6,  sigma=0.03, log_n=False, sign=1.0)
    
    assert_almost_equal(y_, y)

def test_nht_size_nn():
    """ Test nht_size_nn"""
    y1_ = np.array([34158.32460389, 34158.32460389])
    y1 = nht_size_nn(np.sqrt(0.68*(1-0.68)), d=0.01, conf=0.95, power=0.8)
    
    assert_almost_equal(y1_, y1)
    
    y2_ = np.array([339.26886751, 339.26886751])
    y2 = nht_size_nn(np.sqrt(0.68*(1-0.68)), d=0.1, conf=0.95, power=0.8, N=100000)
    
    assert_almost_equal(y2_, y2)
    
    
def test_profit_perfect_nn():
    
    y_ = np.float64(0.5169256875064326)
    y = profit_perfect_nn(mu=0.5, sigma=0.03)
    
    assert_almost_equal(y_, y)
    
def test_error_rate_nn():
    
    y_ = np.float64(0.1445253424807597)
    y = error_rate_nn(n = np.array([2283.89002819, 2283.89002819]), s=np.sqrt(0.68*(1-0.68)), sigma=0.02)
    
    assert_almost_equal(y_, y)
    
def test_profit_nn_sim():
    
    
    y = profit_nn_sim(n=np.array([2283.89002819, 2283.89002819]), N=100000, s=np.sqrt(0.68*(1-0.68)), 
                      mu=0.6,  sigma=0.03, K=2, TS=True, R=10, seed=42)
    