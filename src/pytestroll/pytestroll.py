import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed



#class TestRoll:
#    """
#    Performs various quantities required for test & roll A/B testing
#    "
#    
#    def __init__(self, method: str = 'ips', tau: float = 0.001):
            

def tr_size_nn(N, s, mu=None, sigma=None):
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
    N : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if(type(s) == float):
        s = np.array([s])
    
    if s.size == 1: # symmetric 
        n = ( -3 * s ** 2 + np.sqrt( 9*s**4 + 4*N*s**2*sigma**2 )) / (4 * sigma**2)
        res = np.repeat(n, 2, axis=0)
    
    else:
        n = minimize(profit_nn,
                     args=(N, s, mu, sigma, True, -1.0),
                     x0 = np.log(np.repeat(round(N*0.05), 2 ) - 1),
                     method = 'Nelder-Mead',
                     )['x']
        
        res = np.exp(n)
        
    
    return res 

#np.set_printoptions(suppress=True)
#test_size_nn(N, s, mu, sigma)

# test_size_nn <- function(N, s, mu, sigma) {
#   # computes the profit-maximizing test size for a 2-armed Test & Roll
#   # where response is normal with normal priors (possibly asymmetric)
#   # N is the size of the deployment population
#   # s is a vector of lenght 2 of the (known) std dev of the outcome
#   # mu is a vector of length 2 of the means of the prior on the mean response 
#   # sigma is a vector of length 2 of the std dev of the prior on the mean response
#   # if lenght(s)=1 symmetric priors are assumed and only the first elements of mu and sigma are used
#   stopifnot(N>2, sum(s <= 0) == 0, sum(sigma <= 0) == 0)
#   if (length(s)==1) { # symmetric 
#     n <- ( -3 * s[1]^2 + sqrt( 9*s[1]^4 + 4*N*s[1]^2*sigma[1]^2 ) ) / (4 * sigma[1]^2)
#     n <- rep(n, 2)
#   } else { 
#     n <- optim(par=log(rep(round(N*0.05), 2 ) - 1), fn=profit_nn, control=list(fnscale=-1), 
#                N=N, s=s, mu=mu, sigma=sigma, log_n=TRUE)$par
#     n <- exp(n)
#     }
#   n
# }
     

      
def profit_nn(n, N, s, mu, sigma, log_n=False, sign=1.0):
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
    N : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    log_n : TYPE, optional
        DESCRIPTION. The default is FALSE.

    Returns
    -------
    None.

    """
    
    if log_n:
        n = np.exp(n)
    
    if(type(s) == float):
        s = np.array([s])
        
    if s.size == 1:
        deploy = (N - n[0] - n[1]) * (mu + (2 * sigma**2) / 
                              (np.sqrt(2*np.pi) * np.sqrt(s**2 * (n[0] + n[1]) / (n[0]*n[1]) + 2 * sigma**2))) #Eq.9
        test = mu * (n[0] + n[1]) 
        
    else:
        e = mu[0] - mu[1]
        v = np.sqrt (sigma[0]**4 / (sigma[0]**2 + s[0]**2 / n[0] ) + sigma[1]**4 / (sigma[1]**2 + s[1]**2 / n[1] ) )     
        deploy = (N - n[0] - n[1]) * ( mu[1] + e * norm.cdf(e/v) + v * norm.pdf(e/v) ) #Eq.38
        test = mu[0] * n[0] + mu[1] * n[1] 
          
    res = sign * (deploy + test)/N
        
    return res
        
        
      
# # FUNCTIONS FOR 2-ARM TEST & ROLL (SYMMETRIC AND ASYMMETRIC) =====
# profit_nn <- function(n, N, s, mu, sigma, log_n=FALSE) {
#   # computes the per-customer profit for test & roll with 2 arms
#   # where response is normal with (possibly assymetric) normal priors  
#   # n is a vector of length 2 of sample sizes
#   # N is the size of the deployment population
#   # s is a vector of lenght 2 of the (known) std dev of the outcome
#   # mu is a vector of length 2 of the means of the prior on the mean response 
#   # sigma is a vector of length 2 of the std dev of the prior on the mean response 
#   # if length(n)=1, equal sample sizes are assumed
#   # if lenght(s)=1 symmetric priors are assumed and only the first elements of mu and sigma are used
#   if (length(n) == 1) n <- rep(n, 2)
#   if (log_n) n <- exp(n)
#   stopifnot(N >= sum(n), n[1]>0, n[2]>0, sum(sigma <= 0) == 0, sum(s <= 0) == 0)
#   if (length(s) == 1) { # symmetric
#     deploy <- (N - n[1] - n[2]) * (mu[1] + (2 * sigma[1]^2) / 
#                               (sqrt(2*pi) * sqrt(s[1]^2 * (n[1] + n[2]) / (n[1]*n[2]) + 2 * sigma[1]^2)))
#     test <- mu[1] * (n[1] + n[2]) 
#   } else { # asymmetric
#     e <- mu[1] - mu[2]
#     v <- sqrt (sigma[1]^4 / (sigma[1]^2 + s[1]^2 / n[1] ) + sigma[2]^4 / (sigma[2]^2 + s[2]^2 / n[2] ) )
#     deploy <- (N - n[1] - n[2]) * ( mu[2] + e * pnorm(e/v) + v * dnorm(e/v) )
#     test <- mu[1] * n[1] + mu[2] * n[2] 
#   }
#   #c(profit = test + deploy, profit_test = test, profit_deploy = deploy)
#   (test + deploy)/N
# }

def nht_size_nn(s, d, conf=0.95, power=0.8, N=None):
    """
    
    Parameters
    ----------
    s : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    conf : TYPE, optional
        DESCRIPTION. The default is 0.95.
    power : TYPE, optional
        DESCRIPTION. The default is 0.8.
    N : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if(type(s) == float):
        s = np.array([s])
    
    z_alpha = norm.ppf(1 - (1-conf)/2)
    z_beta = norm.ppf(power)
    
    if s.size == 1:
        if N is None:
            out = (z_alpha + z_beta)**2 * (2 * s**2) / d**2 
        else:
            out = (z_alpha + z_beta)**2 * (2 * s**2) * N /                 \
                  (d**2 * (N-1) + (z_alpha + z_beta)**2 * 4 * s**2)
                  
        res = np.repeat(out, 2, axis=0)
    else:
        if N is None:
            n1 = (z_alpha + z_beta)**2 * (s[0]**2 + s[0]*s[1]) / d**2
            n2 = (z_alpha + z_beta)**2 * (s[0]*s[1] + s[1]**2) / d**2
            
        else:
            n1 = (z_alpha + z_beta)**2 * N * (s[0]**2 + s[0]*s[1]) /       \
                 (d**2 * (N-1) + (z_alpha + z_beta)**2 * (s[0] + s[1])**2)
            n2 = (z_alpha + z_beta)**2 * N * (s[1]**2 + s[0]*s[1]) /        \
                 (d**2 * (N-1) + (z_alpha + z_beta)**2 * (s[0] + s[1])**2)
            
        res = np.array([n1, n2])
        
    return res
    
    

# test_size_nht <- function(s, d, conf=0.95, power=0.8, N=NULL) {
#   # computes the reccomended sample size for a null hypothesis test 
#   # comparing two treatments with finite population correction
#   # s is a vector of length 1 (symmetric) or 2 (asymmetric) 
#   #   indicating repsonse std deviation(s)
#   # d is the minimum detectable difference between treatments
#   # conf is 1 - rate of type I error of the null hypothesis test
#   # power is 1 - rate of type II error of the null hypothesis test
#   # N is the finite population. If N=NULL, then no finite population correction is used.
#   z_alpha <- qnorm(1 - (1-conf)/2)
#   z_beta <- qnorm(power)
#   if(length(s) == 1) { # symmetric response variance
#     if (is.null(N)) {
#       out <- (z_alpha + z_beta)^2 * (2 * s^2) / d^2 
#     } else {
#       out <- (z_alpha + z_beta)^2 * (2 * s^2) * N / 
#              (d^2 * (N-1) + (z_alpha + z_beta)^2 * 4 * s^2)
#     }
#   } else { # asymmetric response variance
#     if(is.null(N)) {
#       n1 <- (z_alpha + z_beta)^2 * (s[1]^2 + s[1]*s[2]) / d^2
#       n2 <- (z_alpha + z_beta)^2 * (s[1]*s[2] + s[2]^2) / d^2
#       out <- c(n1, n2)
#     } else {
#       n1 <- (z_alpha + z_beta)^2 * N * (s[1]^2 + s[1]*s[2]) / 
#             (d^2 * (N-1) + (z_alpha + z_beta)^2 * (s[1] + s[2])^2)
#       n2 <- (z_alpha + z_beta)^2 * N * (s[2]^2 + s[1]*s[2]) / 
#             (d^2 * (N-1) + (z_alpha + z_beta)^2 * (s[1] + s[2])^2)
#       out <- c(n1, n2)
#     }
    
#   }
#   out
# }

def profit_perfect_nn(mu, sigma):

    res = mu + sigma/np.sqrt(np.pi)   #Eq. 13
    
    return res


# profit_perfect_nn <- function(mu, sigma){
#   # computes the per-customer profit with perfect information
#   # where response is normal with symmetric normal priors
#   # todo: adapt to asymmetric case (closed form may not be possible)
#   stopifnot(sigma > 0)
#   (mu + sigma/sqrt(pi))
# }

def error_rate_nn(n, s, sigma):
    
    if(type(s) == float):
        s = np.array([s])
        
    if s.size != 1:   
        raise ValueError("Method only applicable to symmetric priors.")
    
    res =  1/2 - 2*np.arctan(np.sqrt(2)*sigma*np.sqrt(n[0]*n[1] / (n[0] + n[1]) ) / s ) / (2*np.pi)
    
    return res


# error_rate_nn <- function(n, s, sigma) {
#   # computes the rate of incorrect deployments  
#   # where response is normal with symmetric normal priors
#   if (length(n)==1) n <- rep(n, 2)
#   stopifnot(n[1]>0, n[2]>0, s>0, sigma>0)
#   stopifnot(length(s)==1, length(sigma)==1)
#   1/2 - 2*atan( sqrt(2)*sigma*sqrt(n[1]*n[2] / (n[1] + n[2]) ) / s ) / (2*pi)
# }


def one_rep_profit(n, N, s, mu, sigma, K, TS=False):
     # utility function used in profit_nn_sim() to simulate one set of potential outcomes
     m = np.random.normal(mu, sigma, size=K)    
     y = np.random.normal(m, s, size=(N,K))
     
     # perfect information profit
     perfect_info = sum(y[:,np.argmax(m)])
     postmean = np.empty(2)
     
     # test and roll profit with sample sizes n
     # Normal-Normal model https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
     for k in range(K):
         postvar = 1/(1/sigma[k]**2 + n[k]/s[k]**2)
         postmean[k] = postvar*(mu[k]/sigma[k]**2 + sum(y[0:int(np.floor(n[k])), k])/s[k]**2)
  
     delta = np.argmax(postmean) # pick the arm with the highest posterior mean
     error = delta != np.argmax(m)
     deploy_1 = delta == 0
     test_roll = 0
     for k in range(K):
         test_roll = test_roll + sum(y[0:int(np.floor(n[k])), k]) # profit from first n observations for each arm
     test_roll = test_roll + sum(y[int(np.floor((sum(n)))):N, delta]) # profit from remaining observations for selected arm
     
     # Thompson sampling profit
     thom_samp = None
     if TS:
         n = np.repeat(0, K) # Initialize each arm with zero observations 
         # note mu and sigma are initialized at the priors
         postvar = (1/(np.expand_dims(1/sigma**2, 1) + (np.repeat(np.arange(N), K).reshape((N, K)) + 1).T * np.expand_dims(1/s**2, 1))).T
         postmean = postvar * ((np.expand_dims(mu/sigma**2, 1) + (np.apply_along_axis(np.cumsum, 0, y)).T * np.expand_dims(1/s**2, 1))).T

         for i in range(N):
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




# FUNCTIONS FOR K-ARM TEST & ROLL (REQUIRES SIMULATION) =====
# one_rep_profit <-function(n, N, s, mu, sigma, K, TS=FALSE) {
#   # utility function used in profit_nn_sim() to simulate one set of potential outcomes
#   m <- rnorm(K, mu, sigma) # draw a true mean for the arm
#   y <- matrix(rnorm(N*K, m, s), nrow=N, ncol=K, byrow=TRUE) # N observations from each arm
  
#   # perfect information profit
#   perfect_info <- sum(y[,which.max(m)]) # Perfect information: sum observations from arm with highest m
  
#   # test and roll profit with sample sizes n
#   postmean <- rep(NA, K)
#    for (k in 1:K) {
#     postvar <- 1/(1/sigma[k]^2 + n[k]/s[k]^2)
#     postmean[k] <- postvar*(mu[k]/sigma[k]^2 + sum(y[1:n[k], k])/s[k]^2)
#   }
#   delta <- which.max(postmean) # pick the arm with the highest posterior mean
#   error <- delta != which.max(m)
#   deploy_1 <- delta == 1
#   test_roll <- 0
#   for (k in 1:K) test_roll <- test_roll + sum(y[1:n[k], k]) # profit from first n observations for each arm
#   test_roll <- test_roll + sum(y[(sum(n)+1):N, delta]) # profit from remaining observations for selected arm
  
#   # Thompson sampling profit
#   thom_samp <- NA
#   if (TS==TRUE) {
#     n <- rep(0, K) # Initialize each arm with zero observations 
#     # note mu and sigma are initialized at the priors
#     postvar <- t(1/(1/sigma^2 + t(matrix(1:N, nrow=N, ncol=K)) * 1/s^2))
#     postmean <- postvar * t(mu/sigma^2 + t(apply(y, 2, cumsum)) * 1/s^2)
#     for (i in 1:N) { 
#       k <- which.max(rnorm(K, mu, sigma)) # draw a sample from the current posterior for each arm and choose arm with largest draw
#       n[k] <- n[k] + 1 # increase the number of observations used from sampled arm
#       mu[k] <- postmean[n[k], k] # advance mu and sigma 
#       sigma[k] <- sqrt(postvar[n[k], k])
#     } 
#     thom_samp <- 0
#     for (k in 1:K) thom_samp <- thom_samp + sum(y[1:n[k], k])
#   }
#   return(c(perfect_info=perfect_info, test_roll=test_roll, thom_samp=thom_samp, 
#            error = error, deploy_1=deploy_1))
# }


def profit_nn_sim(n, N, s, mu, sigma, K=2, TS=False, R=1000, seed=42):
    """
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
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
    
    if(type(s) == float):
        s = np.array([s])
    if(type(n) == float):
        n = np.array([n])
        
    if s.size == 1:   
        s = np.repeat(s, K)
        mu = np.repeat(mu, K)
        sigma = np.repeat(sigma, K)
        
    if n.size == 1:
        n = np.repeat(n, K)
    
    perfect_info, test_roll, thom_samp, error, deploy_1 = zip(*Parallel(n_jobs=-2)(delayed(one_rep_profit)(n, N, s, mu, sigma, K, TS) for i in range(R)))
    reps = pd.DataFrame({'perfect_info': perfect_info, 'test_roll': test_roll, 'thom_samp': thom_samp, 'error': error, 'deploy_1': deploy_1})
    
    if not TS:
        reps['thom_samp'] = np.nan
         
    profit = np.vstack([np.apply_along_axis(np.nanmean, 0, reps.iloc[:,0:3].values), 
                        np.apply_along_axis(np.nanquantile, 0, reps.iloc[:,0:3].values, q=np.array([0.05, 0.95]))]) / N
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
    
 

# profit_nn_sim <- function(n, N, s, mu, sigma, K=2, TS=FALSE, R=1000) {
#   # computes the per-customer profit for test & roll with K arms
#   # where response is normal with (assymetric) normal priors 
#   # R is the number of simulation replications
#   # n is the sample size for test & roll
#   # N is the size of the total population
#   # s are the (known) std devs of the outcome (vector of length 1 or K)
#   # mu are the means of the priors on the mean response of length (vector of length 1 or K)
#   # sigma are the std devs of the priors on the mean response (vector of length 1 or K)
#   # TS is a switch for computing profit for Thompson Sampling
#   # if s is length 1, then arms are assumed to be symmetric
#   if (length(s) == 1) { s <- rep(s, K); mu <- rep(mu, K); sigma <- rep(sigma, K) }
#   if (length(n) == 1) n <- rep(n, K)
#   stopifnot(length(s) == K, length(mu) == K, length(sigma) == K)
#   stopifnot(sum(sigma <= 0) == 0, sum(s <= 0) == 0) 
#   stopifnot(N > K)
#   reps <- foreach(i=1:R) %dopar% one_rep_profit(n, N, s, mu, sigma, K, TS)
#   reps <- as.data.frame(do.call(rbind, reps))
#   profit <- apply(reps[,1:3], 2, mean)
#   profit <- rbind(exp_profit=profit, 
#                   apply(reps[,1:3], 2, quantile, probs=c(0.05, 0.95), na.rm=TRUE))
#   profit <- profit / N
#   regret_draws <- 1 - reps[,1:3] / reps$perfect_info
#   regret <- apply(regret_draws, 2, mean)
#   regret <- rbind(exp_regret=regret, 
#                   apply(regret_draws, 2, quantile, probs=c(0.05, 0.95), na.rm=TRUE))
#   error <- mean(reps[,4])
#   deploy_1 <- mean(reps[,5])
#   return(list(profit=profit, regret=regret, error_rate=error, deploy_1_rate = deploy_1,
#               profit_draws=reps, regret_draws=regret_draws))
# }


def test_eval_nn(n, N, s, mu, sigma):
    
     profit = profit_nn(n, N, s, mu, sigma)*N
 
     if(type(s) == float):
        s = np.array([s])
    
     if s.size == 1:
         test = mu * (n[0] + n[1]) 
         deploy = profit - test
         rand = mu*N # choose randomly
         perfect = profit_perfect_nn(mu, sigma)*N 
         error_rate = error_rate_nn(n, s, sigma)
         deploy_1_rate = 0.5
        
     else:
         test = mu[0] * n[0] + mu[1] *n[1] 
         deploy = profit - test
         rand = ((mu[0] + mu[1])*0.5)*N
         out = profit_nn_sim(n, N, s, mu, sigma, R=10000)
         perfect = out['profit'].iloc[0,1]*N
         error_rate = out['error_rate']
         deploy_1_rate =  out['deploy_1_rate']
         
     gain = (profit - rand) / (perfect - rand)
     
     data = {
          'n1':n[0], 
          'n2':n[1],
          'profit_per_cust' : profit/N,
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
        
 
        
# test_eval_nn <- function(n, N, s, mu, sigma) {
#   # provides a complete summary of a test & roll plan
#   # n is a vector of length 2 of sample sizes
#   # N is the size of the deployment population
#   # s is a vector of lenght 2 of the (known) std dev of the outcome
#   # mu is a vector of length 2 of the means of the prior on the mean response 
#   # sigma is a vector of length 2 of the std dev of the prior on the mean response 
#   # if length(n)=1, equal sample sizes are assumed
#   # if lenght(s)=1 symmetric priors are assumed and only the first elements of mu and sigma are used
#   stopifnot(N >= n[1] + n[2], n[1] > 0, n[2] >0, sum(s <= 0) == 0, sum(sigma <=0) == 0)
#   profit <- profit_nn(n, N, s, mu, sigma)*N
#   if (length(s)==1) { # symmetric
#     test <- mu[1] * (n[1] + n[2]) 
#     deploy <- profit - test
#     rand <- mu[1]*N # choose randomly
#     perfect <- profit_perfect_nn(mu, sigma)*N 
#     error_rate <- error_rate_nn(n, s, sigma)
#     deploy_1_rate <- 0.5
#   } else { # assymetric
#     test <- mu[1] * n[1] + mu[2] *n[2] 
#     deploy <- profit - test
#     rand <- ((mu[1] + mu[2])*0.5)*N
#     out <- profit_nn_sim(n, N, s, mu, sigma, R=10000)
#     perfect <- out$profit["exp_profit", "perfect_info"]*N
#     error_rate <- out$error_rate
#     deploy_1_rate <- out$deploy_1_rate
#   }
#   gain <- (profit - rand) / (perfect - rand)
#   data.frame(n1=n[1], n2=n[2],
#              profit_per_cust = profit/N,
#              profit = profit, 
#              profit_test = test, 
#              profit_deploy = deploy,
#              profit_rand = rand,
#              profit_perfect = perfect, 
#              profit_gain = gain, 
#              regret = 1-profit/perfect,
#              error_rate = error_rate, 
#              deploy_1_rate = deploy_1_rate, 
#              tie_rate = 0)
# }


def one_rep_test_size(n_vals, N, s, mu, sigma, K):
    """

    Parameters
    ----------
    n_vals : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    m = np.random.normal(mu, sigma, size=K)  
    y = np.random.normal(m, s, size=(N,K))
     
    postmean = np.empty(shape=(len(n_vals), K))
    for k in range(K):
        postvar = 1/(1/sigma[k]**2 + (n_vals+1)/s[k]**2)
        postmean[:,k] = postvar*(mu[k]/sigma[k]**2 + np.cumsum(y[0:int(np.floor(N/K)-1),k])/s[k]**2)
      
    delta = np.argmax(postmean, axis=1) # pick the arm with the highest posterior mean
    #error = delta != np.argmax(m)
    profit = np.repeat(0.0, len(n_vals))
    for i in n_vals:
        for k in range(K): profit[i] = profit[i] + np.sum(y[:n_vals[i], k]) # profit from first n[i] observations for each arm
        profit[i] = profit[i] + np.sum(y[(n_vals[i]*K):N, delta[i]]) # profit from remaining observations for selected arm
    
    return n_vals, profit
        

# one_rep_test_size <- function(n_vals, N, s, mu, sigma, K) {
#   # utility function used in test_size_nn_sim() to simulate one set of potential outcomes
#   # and profits for all possible equal sample sizes
  
#   # potential outcomes
#   m <- rnorm(K, mu, sigma) # draw a true mean for the arm
#   y <- matrix(rnorm(N*K, m, s), nrow=N, ncol=K, byrow=TRUE) # N observations from each arm
  
#   postmean <- matrix(NA, nrow=length(n_vals), ncol=K)
#   for (k in 1:K) {
#     postvar <- 1/(1/sigma[k]^2 + n_vals/s[k]^2)
#     postmean[,k] <- postvar*(mu[k]/sigma[k]^2 + cumsum(y[1:(floor(N/K)-1),k])/s[k]^2)
#   }
#   delta <- apply(postmean, 1, which.max) # pick the arm with the highest posterior mean
#   error <- delta != which.max(m)
#   profit <- rep(0, length(n_vals))
#   for (i in seq_along(n_vals)) {
#     for (k in 1:K) profit[i] <- profit[i] + sum(y[1:n_vals[i], k]) # profit from first n[i] observations for each arm
#     profit[i] <- profit[i] + sum(y[(n_vals[i]*K + 1):N, delta[i]]) # profit from remaining observations for selected arm
#   }
#   return(cbind(n=n_vals, profit))
# }



def test_size_nn_sim(N, s, mu, sigma, K=2, R=1000):
    
    if(type(s) == float):
        s = np.array([s])
      
    if s.size == 1:   
    
        s = np.repeat(s, K)
        mu = np.repeat(mu, K)
        sigma = np.repeat(sigma, K)
    
        n_vals = np.array(range(int(np.floor(N/K)-1)))
    
        n_vals_, profit_ = zip(*Parallel(n_jobs=-2)(delayed(one_rep_test_size)(n_vals, N, s, mu, sigma, K) for i in range(R)))
        reps = pd.DataFrame({'n_vals': np.hstack(n_vals_), 'profit': np.hstack(profit_)})
        exp_profit = pd.pivot_table(reps, values='profit', columns = 'n_vals', aggfunc = 'sum') / R
        n = np.repeat(n_vals[np.argmax(exp_profit)], K)
        max_exp_profit = np.max(exp_profit.values)/N
    
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

# test_size_nn_sim <- function(N, s, mu, sigma, K=2, R=1000) {
#   # computes the profit-maximizing test size for a multi-armed test & roll
#   # where response is normal with normal priors (possibly asymmetric)
#   # N is the size of the deployment population
#   # K is the number of arms
#   # s is a K-vector of (known) std devs of the response, if length(s)==1, symmetric priors are used
#   # mu is a K-vector of length K the means of the priors on the outcome 
#   # sigma is a K-vector of std devs of the priors on the mean response
#   stopifnot(N > 2, sum(s <= 0) == 0, sum(sigma <= 0) == 0)
#   if (length(s==1)) { # symmetric amrs
#     # n is same for all arms; solve by enumeration
#     s <- rep(s, K); mu <- rep(mu, K); sigma <- rep(sigma, K)
#     n_vals <- 1:(floor(N/K)-1) # potential values for n
#     reps <- foreach(i=1:R) %dopar% one_rep_test_size(n_vals, N, s, mu, sigma, K)
#     reps <- as.data.frame(do.call(rbind, reps))
#     exp_profit <- xtabs(profit ~ n, data=reps) / R
#     n <- rep(n_vals[which.max(exp_profit)], K)
#   } else { # asymmetric
#     stopifnot(length(mu)==K, length(s) == K, length(sigma) == K)
#     # todo:finish this
#     # best option is to use a numeric optimization, but then noise in profit_nn_sim
#     # becomes critical. comments out is one strategy
#     # start values based on two-arm symmetric
#     #n <- ( -3 * s[1]^2 + sqrt( 9*s[1]^4 + 4*N*s[1]^2*sigma[1]^2 ) ) / (4 * sigma[1]^2)
#     #n <- optim(par=log(n), fn=profit_nn_sim, control=list(fnscale=-1), 
#     #           N=N, s=s, mu=mu, sigma=sigma, K=K, R=1000, log_n=TRUE)$par
#     #n <- optim(par=n, fn=profit_nn_sim, control=list(fnscale=-1),  # more precise simulation
#     #           N=N, s=s, mu=mu, sigma=sigma, K=K, R=5000, log_n=TRUE)$par
#     n <- NA
#   }
#   return(list(n=n, max(exp_profit)/N))
# }
