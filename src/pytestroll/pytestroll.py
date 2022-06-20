import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed


class NHST:
    def __init__(self, s:float = None, d:float = None, 
                       conf:float = 0.95, power:float = 0.8) -> None:
        
        if(type(s) == float):
            s = np.array(s)
        self.s = s
        self.d = d
        self.conf = conf
        self.power = power
     
    
    def __repr__(self) -> str:
            
            items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))


    def nht_size_nn(self, N:int = None):
        """
        
        Parameters
        ----------
       
        Returns
        -------
        None.

        """
        
        z_alpha = norm.ppf(1 - (1-self.conf)/2)
        z_beta = norm.ppf(self.power)
        
        if self.s.size == 1:
            if N is None:
                out = (z_alpha + z_beta)**2 * (2 * self.s**2) / self.d**2 
            else:
                out = (z_alpha + z_beta)**2 * (2 * self.s**2) * N /                          \
                      (self.d**2 * (N-1) + (z_alpha + z_beta)**2 * 4 * self.s**2)
                      
            res = np.repeat(out, 2, axis=0)
        else:
            if N is None:
                n1 = (z_alpha + z_beta)**2 * (self.s[0]**2 + self.s[0]*self.s[1]) / self.d**2
                n2 = (z_alpha + z_beta)**2 * (self.s[0]*self.s[1] + self.s[1]**2) / self.d**2
                
            else:
                n1 = (z_alpha + z_beta)**2 * N * (self.s[0]**2 + self.s[0]*self.s[1]) /            \
                     (self.d**2 * (N-1) + (z_alpha + z_beta)**2 * (self.s[0] + self.s[1])**2)
                n2 = (z_alpha + z_beta)**2 * N * (self.s[1]**2 + self.s[0]*self.s[1]) /        \
                     (self.d**2 * (N-1) + (z_alpha + z_beta)**2 * (self.s[0] + self.s[1])**2)
                
            res = np.array([n1, n2])
            
        self.sample_size = res
            
        return self.sample_size
        

class TestRoll:
    """
    
    Test Roll Class
         
    """

    def __init__(self, N:int = None, s:float = None, mu:float = None, sigma:float = None) -> None:
        
        if(type(s) == float):
            s = np.array(s)
            
        self.N = N
        self.s = s
        self.mu = mu
        self.sigma = sigma
        
    def __repr__(self) -> str:
            
            items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))

 
    def tr_size_nn(self):
        """
        # computes the profit-maximizing test size for a 2-armed Test & Roll
        # where response is normal with normal priors (possibly asymmetric)
        # N is the size of the deployment population
        # s is a vector of lenght 2 of the (known) std dev of the outcome
        # mu is a vector of length 2 of the means of the prior on the mean response 
        # sigma is a vector of length 2 of the std dev of the prior on the mean response
        # if lenght(s)=1 symmetric priors are assumed and only the first elements of mu and sigma are used

        Parameters
        ----------
        
        Returns
        -------
        None.
        
        """
         
        if self.s.size == 1: # symmetric 
            n = ( -3 * self.s ** 2 + np.sqrt( 9*self.s**4 + 4*self.N*self.s**2*self.sigma**2 )) / (4 * self.sigma**2)
            self.sample_size = np.repeat(n, 2, axis=0)
        
        else:
            n = minimize(self.profit_nn,
                         args=(True, -1.0),
                         x0 = np.log(np.repeat(round(self.N*0.05), 2 ) - 1),
                         method = 'Nelder-Mead',
                         )['x']
            
            self.sample_size = np.exp(n)
            
        return self.sample_size 
     
    def profit_nn(self, n:float = None, log_n:bool = False, sign:float = 1.0):
        """
        # computes the per-customer profit for test & roll with 2 arms
        # where response is normal with (possibly assymetric) normal priors  
        # n is a vector of length 2 of sample sizes
        # N is the size of the deployment population
        # s is a vector of lenght 2 of the (known) std dev of the outcome
        # mu is a vector of length 2 of the means of the prior on the mean response 
        # sigma is a vector of length 2 of the std dev of the prior on the mean response 
        # if length(n)=1, equal sample sizes are assumed
        # if lenght(s)=1 symmetric priors are assumed and only the first elements of mu and sigma are used

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        log_n : TYPE, optional
            DESCRIPTION. The default is FALSE.

        Returns
        -------
        None.

        """
        
        if log_n:
            n = np.exp(n)
        
        if self.s.size == 1:
            deploy = (self.N - n[0] - n[1]) * (self.mu + (2 * self.sigma**2) / 
                                  (np.sqrt(2*np.pi) * np.sqrt(self.s**2 * (n[0] + n[1]) / (n[0]*n[1]) + 2 * self.sigma**2))) #Eq.9
            test = self.mu * (n[0] + n[1]) 
            
        else:
            e = self.mu[0] - self.mu[1]
            v = np.sqrt (self.sigma[0]**4 / (self.sigma[0]**2 + self.s[0]**2 / n[0] ) + 
                         self.sigma[1]**4 / (self.sigma[1]**2 + self.s[1]**2 / n[1] ) )     
            deploy = (self.N - n[0] - n[1]) * (self.mu[1] + e * norm.cdf(e/v) + v * norm.pdf(e/v) ) #Eq.38
            test = self.mu[0] * n[0] + self.mu[1] * n[1] 
              
        res = sign * (deploy + test)/self.N
            
        return res
    
    def _one_rep_profit(self, n:float = None, K:int = None, TS:bool = False, seed:int = None):
        
         np.random.seed(seed)
         
         # utility function used in profit_nn_sim() to simulate one set of potential outcomes
         m = np.random.normal(self.mu, self.sigma, size=K)    
         y = np.random.normal(m, self.s, size=(self.N,K))
         
         # perfect information profit
         perfect_info = sum(y[:,np.argmax(m)])
         postmean = np.empty(2)
         
         # test and roll profit with sample sizes n
         # Normal-Normal model https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
         for k in range(K):
             postvar = 1/(1/self.sigma[k]**2 + n[k]/self.s[k]**2)
             postmean[k] = postvar*(self.mu[k]/self.sigma[k]**2 + sum(y[0:int(np.floor(n[k])), k])/self.s[k]**2)
      
         delta = np.argmax(postmean) # pick the arm with the highest posterior mean
         error = delta != np.argmax(m)
         deploy_1 = delta == 0
         test_roll = 0
         for k in range(K):
             test_roll = test_roll + sum(y[0:int(np.floor(n[k])), k]) # profit from first n observations for each arm
         test_roll = test_roll + sum(y[int(np.floor((sum(n)))):self.N, delta]) # profit from remaining observations for selected arm
         
         # Thompson sampling profit
         thom_samp = None
         if TS:
             n = np.repeat(0, K) # Initialize each arm with zero observations 
             # note mu and sigma are initialized at the priors
             postvar = (1/(np.expand_dims(1/self.sigma**2, 1) + (np.repeat(np.arange(self.N), K).reshape((self.N, K)) + 1).T * np.expand_dims(1/self.s**2, 1))).T
             postmean = postvar * ((np.expand_dims(self.mu/self.sigma**2, 1) + (np.apply_along_axis(np.cumsum, 0, y)).T * np.expand_dims(1/self.s**2, 1))).T

             mu = self.mu
             sigma = self.sigma
            
             for i in range(self.N):
                 k = np.argmax(np.random.normal(mu, sigma, size=K))
                 mu[k] = postmean[n[k], k] # advance mu and sigma 
                 sigma[k] = np.sqrt(postvar[n[k], k])
                 n[k] =n[k]+ 1 # increase the number of observations used from sampled arm
             thom_samp = 0
             for k in range(K): 
                 thom_samp = thom_samp + sum(y[:n[k], k])
         #return {"perfect_info":perfect_info, "test_roll":test_roll, "thom_samp":thom_samp, 
         #            "error" : error, "deploy_1":deploy_1}
         return perfect_info, test_roll, thom_samp, error, deploy_1


    def profit_nn_sim(self, n:float = None, K:int = 2, TS:bool = False, R:int = 1000, seed:int = 42):
        """

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
       K : TYPE, optional
            DESCRIPTION. The default is 2.
        TS : TYPE, optional
            DESCRIPTION. The default is FALSE.
        R : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.

        """
        
        np.random.seed(seed)
        r = [np.random.randint(1,10000) for _ in range(R)]
        
        if(type(n) == float):
            n = np.array([n])
            
        s = self.s
        mu = self.mu
        sigma = self.sigma
            
        if s.size == 1:   
            self.s = np.repeat(s, K)
            self.mu = np.repeat(mu, K)
            self.sigma = np.repeat(sigma, K)
             
        if n.size == 1:
            n = np.repeat(n, K)
        
        perfect_info, test_roll, thom_samp, error, deploy_1 = zip(*Parallel(n_jobs=-2)(delayed(self._one_rep_profit)(n, K, TS, seed=r[i]) for i in range(R)))
        reps = pd.DataFrame({'perfect_info': perfect_info, 'test_roll': test_roll, 'thom_samp': thom_samp, 'error': error, 'deploy_1': deploy_1})
        
        if not TS:
            reps['thom_samp'] = np.nan
             
        profit = np.vstack([np.apply_along_axis(np.nanmean, 0, reps.iloc[:,0:3].values), 
                            np.apply_along_axis(np.nanquantile, 0, reps.iloc[:,0:3].values, q=np.array([0.05, 0.95]))]) / self.N
        profit = pd.DataFrame(profit, columns=['perfect_info', 'test_roll', 'TS'])
        profit.insert(0, '', ['mean_profit', '5%', '95%'])  

        regret_draws = 1 - reps.iloc[:,0:3].values / np.expand_dims(reps.iloc[:,0].values, 1)
        regret_draws = pd.DataFrame(regret_draws, columns=['perfect_info', 'test_roll', 'thom_samp'])
        regret = np.vstack([np.apply_along_axis(np.nanmean, 0, regret_draws),
                           np.apply_along_axis(np.nanquantile, 0, regret_draws, q=np.array([0.05, 0.95]))]
                           )
        regret = pd.DataFrame(regret, columns=['perfect_info', 'test_roll', 'TS'])
        regret.insert(0, '', ['mean_profit', '5%', '95%'])  
        
        error = np.mean(reps['error'].values)
        deploy_1 = np.mean(reps['deploy_1'].values)
        
        return {'profit':profit, 'regret':regret, 'error_rate':error, 'deploy_1_rate':deploy_1,
                'profit_draws':reps, 'regret_draws':regret_draws}
    
    
    def _profit_perfect_nn(self):

        res = self.mu + self.sigma/np.sqrt(np.pi)   #Eq. 13
        
        return res



    def _error_rate_nn(self, n:float = None):
        
        if self.s.size != 1:   
            raise ValueError("Method only applicable to symmetric priors.")
        
        res =  1/2 - 2*np.arctan(np.sqrt(2)*self.sigma*np.sqrt(n[0]*n[1] / (n[0] + n[1]) ) / self.s ) / (2*np.pi)
        
        return res

    
    
    def eval_nn(self, n:float = None, R:int = None):
        
         profit = self.profit_nn(n)*self.N
     
         if self.s.size == 1:
             test = self.mu * (n[0] + n[1]) 
             deploy = profit - test
             rand = self.mu * self.N # choose randomly
             perfect = self._profit_perfect_nn() * self.N 
             error_rate = self._error_rate_nn(n)
             deploy_1_rate = 0.5
            
         else:
             test = self.mu[0] * n[0] + self.mu[1] *n[1] 
             deploy = profit - test
             rand = ((self.mu[0] + self.mu[1])*0.5)*self.N
             out = self.profit_nn_sim(n, R=R)
             perfect = out['profit'].iloc[0,1]*self.N
             error_rate = out['error_rate']
             deploy_1_rate =  out['deploy_1_rate']
             
         gain = (profit - rand) / (perfect - rand)
         
         data = {
              'n1':n[0], 
              'n2':n[1],
              'profit_per_cust' : profit/self.N,
              'profit': profit, 
              'profit_test' : test, 
              'profit_deploy' : deploy,
              'profit_rand' : rand,
              'profit_perfect' : perfect, 
              'profit_gain' : gain, 
              'regret' : 1-profit/perfect,
              'error_rate' : error_rate, 
              'deploy_1_rate' : deploy_1_rate, 
              'tie_rate' : 0.0
             }
         return data 
            
     
    def _one_rep_test_size(self, n_vals:float = None, K:int = None, seed:int = None):
         """

         Parameters
         ----------
         n_vals : TYPE
             DESCRIPTION.
        K : TYPE
             DESCRIPTION.

         Returns
         -------
         None.

         """
         np.random.seed(seed)
         
         m = np.random.normal(self.mu, self.sigma, size=K)  
         y = np.random.normal(m, self.s, size=(self.N,K))
          
         postmean = np.empty(shape=(len(n_vals), K))
         for k in range(K):
             postvar = 1/(1/self.sigma[k]**2 + (n_vals+1)/self.s[k]**2)
             postmean[:,k] = postvar*(self.mu[k]/self.sigma[k]**2 + np.cumsum(y[0:int(np.floor(self.N/K)-1),k])/self.s[k]**2)
           
         delta = np.argmax(postmean, axis=1) # pick the arm with the highest posterior mean
         #error = delta != np.argmax(m)
         profit = np.repeat(0.0, len(n_vals))
         for i in n_vals:
             for k in range(K): profit[i] = profit[i] + np.sum(y[:n_vals[i], k]) # profit from first n[i] observations for each arm
             profit[i] = profit[i] + np.sum(y[(n_vals[i]*K):self.N, delta[i]]) # profit from remaining observations for selected arm
         
         return n_vals, profit
             


    def tr_size_nn_sim(self, K:int = 2, R:int = 1000, seed:int = 42):
         
         np.random.seed(seed)
         r = [np.random.randint(1,10000) for _ in range(R)]
         
         s = self.s
         mu = self.mu
         sigma = self.sigma
           
         if s.size == 1:   
         
             self.s = np.repeat(s, K)
             self.mu = np.repeat(mu, K)
             self.sigma = np.repeat(sigma, K)
         
             n_vals = np.array(range(int(np.floor(self.N/K)-1)))
         
             n_vals_, profit_ = zip(*Parallel(n_jobs=-2)(delayed(self._one_rep_test_size)(n_vals, K, seed=r[i]) for i in range(R)))
             reps = pd.DataFrame({'n_vals': np.hstack(n_vals_), 'profit': np.hstack(profit_)})
             exp_profit = pd.pivot_table(reps, values='profit', columns = 'n_vals', aggfunc = 'sum') / R
             n = np.repeat(n_vals[np.argmax(exp_profit)], K)
             max_exp_profit = np.max(exp_profit.values)/self.N
         
         else:
             raise ValueError("Asymmetric test not implemented yet.")
             
             #n = None
             # todo:finish this
             # best option is to use a numeric optimization, but then noise in profit_nn_sim
             # becomes critical. comments out is one strategy
             # start values based on two-arm symmetric
             #n <- ( -3 * s[1]^2 + sqrt( 9*s[1]^4 + 4*N*s[1]^2*sigma[1]^2 ) ) / (4 * sigma[1]^2)
             #n <- optim(par=log(n), fn=profit_nn_sim, control=list(fnscale=-1), 
             #           N=N, s=s, mu=mu, sigma=sigma, K=K, R=1000, log_n=TRUE)$par
             #n <- optim(par=n, fn=profit_nn_sim, control=list(fnscale=-1),  # more precise simulation
             #           N=N, s=s, mu=mu, sigma=sigma, K=K, R=5000, log_n=TRUE)$par
             
         return {'n':n, 'max_exp_profit':max_exp_profit}
  
     
            