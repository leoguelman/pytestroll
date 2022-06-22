from pytestroll.pytestroll import (
    NHST,
    TestRoll
    )

from numpy.testing import assert_almost_equal
import numpy as np


def test_tr_size_nn():
    """ Test tr_size_nn"""
    
    # Symmetric
    y1_ = np.array([2283.89002819, 2283.89002819])
    y1 = TestRoll(N = 100000, s=np.sqrt(0.68*(1-0.68)), mu=0.68, sigma=0.03).tr_size_nn()
    
    # Asymmetric 
    y2_ = np.array([1858.50643038, 92412.762479])
    y2 = TestRoll(N = 10000000, s=np.array([0.4, 0.6]), mu=np.array([0.5, 0.6]), sigma=np.array([0.03,0.03])).tr_size_nn()
    
    assert_almost_equal(y1_, y1)
    assert_almost_equal(y2_, y2)


def test_profit_nn():
    """ Test profit_nn"""
    
    y_ = np.float64(0.6153599861754739)
    
    y = TestRoll(N=100000, s=np.sqrt(0.68*(1-0.68)), mu=0.60, sigma=0.03).profit_nn(n=np.array([2283.89002819, 2283.89002819]))
     
    assert_almost_equal(y_, y)

def test_nht_size_nn():
    """ Test nht_size_nn"""
    
    # Symmetric
    y1_ = np.array([34158.32460389, 34158.32460389])
    y1 = NHST(s=np.sqrt(0.68*(1-0.68)), d=0.01, conf=0.95, power=0.8).nht_size_nn()
      
    assert_almost_equal(y1_, y1)
    
    # Symmetric (with finite population correction)
    y2_ = np.array([339.26886751, 339.26886751])
    y2 = NHST(s=np.sqrt(0.68*(1-0.68)), d=0.1, conf=0.95, power=0.8).nht_size_nn(N=100000)
    
    assert_almost_equal(y2_, y2)
    
    # Asymmetric
    y3_ = np.array([431.68838539, 518.02606247])
    y3 = NHST(s=np.array([0.5, 0.6]), d=0.1, conf=0.95, power=0.8).nht_size_nn()
    
    assert_almost_equal(y3_, y3)
    
    # Asymmetric (with finite population correction)
    y4_ = np.array([427.63138466, 513.15766159])
    y4 = NHST(s=np.array([0.5, 0.6]), d=0.1, conf=0.95, power=0.8).nht_size_nn( N=100000)
    
    assert_almost_equal(y4_, y4)
    
def test_profit_perfect_nn():
    
    y_ = np.float64(0.5169256875064326)
    y = TestRoll(N=100000, s=np.sqrt(0.68*(1-0.68)), mu=0.5, sigma=0.03)._profit_perfect_nn()
    
    assert_almost_equal(y_, y)
    
def test_error_rate_nn():
    
    y_ = np.float64(0.1445253424807597)
    
    y = TestRoll(N=100000, s=np.sqrt(0.68*(1-0.68)), mu=0.5, sigma=0.02)._error_rate_nn(n = np.array([2283.89002819, 2283.89002819]))
    
    assert_almost_equal(y_, y)
    
def test_profit_nn_sim():
    
    # Symmetric
    yp = np.array([0.6128922297385954, 0.6094488069018906, 0.6115283928708336])
    yr = np.array([0.0, 0.005651146214613689, 0.0022205550641044014])
    ye = np.float64(0.4)
    yd1 = np.float64(0.6)
    
    y1 = TestRoll(N=100000, s=np.sqrt(0.68*(1-0.68)), mu=0.6, sigma=0.03
                  ).profit_nn_sim(n=np.array([2283.89002819, 2283.89002819]),
                                  K=2, TS=True, R=10, seed=42             
                                  )
    assert_almost_equal(yp, y1['profit'].iloc[0][1:].values)
    
    assert_almost_equal(yr, y1['regret'].iloc[0][1:].values)
        
    assert_almost_equal(ye, y1['error_rate'])
    
    assert_almost_equal(yd1, y1['deploy_1_rate'])
    
    
    # Asymmetric
    yp = np.array([0.7668313399131115, 0.75801587125134, 0.7662130662698954])
    yr = np.array([0.0, 0.016836104142319997, 0.0009021622750046254])
    ye = np.float64(0.1)
    yd1 = np.float64(0.4)
    
    y2 = TestRoll(N=100000, s=np.array([0.5, 0.6]), 
                  mu=np.array([0.5, 0.6]),  sigma=np.array([0.5, 0.6])
                  ).profit_nn_sim(n=np.array([1000, 2000]),
                                  K=2, TS=True, R=10, seed=42             
                                                                                        )  
    
    assert_almost_equal(yp, y2['profit'].iloc[0][1:].values)
    
    assert_almost_equal(yr, y2['regret'].iloc[0][1:].values)
        
    assert_almost_equal(ye, y2['error_rate'])
    
    assert_almost_equal(yd1, y2['deploy_1_rate'])
    
    
    
def test_tr_size_nn_sim():
    
    # Symmetric
    y1_ = {'n': np.array([590, 590]), 'max_exp_profit': 0.6117075409508286}
    
    y1 = TestRoll(N=100000, s=np.sqrt(0.68*(1-0.68)), mu=0.6, sigma=0.03
                  ).tr_size_nn_sim(K=2, R=10, seed=42
                                   ) 
    assert_almost_equal(y1_['n'], y1['n'])
    
    assert_almost_equal(y1_['max_exp_profit'], y1['max_exp_profit'])
    
    
    # Asymmetric -- Not implemented yet