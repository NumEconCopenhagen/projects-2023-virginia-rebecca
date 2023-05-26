import numpy as np
from scipy import optimize
from types import SimpleNamespace
import matplotlib.pyplot as plt


class Problem2 : 

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.eta = 0.5
        par.w = 1.0
        par.k = 1.0
        par.rho = 0.90
        par.iota = 0.01
        par.sigmae = 0.10
        par.R = (1.0 + 0.01)**(1/12)
        par.T = 120
        par.delta = 0.05
    
    # define functions

    def obj(self, l):
        """ profit function """

        par = self.par

        return par.k*l**(1 - par.eta) - par.w*l
    
    def optimal_labor2(self):
        """ calculate optimal l numerically """

        par = self.par

        obj1 = lambda x: -self.obj(x)
        res = optimize.minimize_scalar(obj1)
        l = res.x

        return l 
    
    def foc(self):
        """ calculate l from FOC """

        par = self.par

        return (((1-par.eta)*par.k)/par.w)**(1/par.eta)
    
    def calc_k(self, shock_series): 
        """ calculate ex post value of the salon with optimal policy
        
        Args: 
            self: instance of the class
            shock series: random series of shocks
        
        Returns:
            h: ex post value of the salon
        
        """   
        par = self.par
        
        # prepare arrays
        kkappa = np.zeros(par.T) # demand shocks
        ll = np.zeros(par.T) # employees
        profit = np.zeros(par.T) # profits
        
        # set initial salon value
        salon_value = 0.0
        
        # loop to fill vectors with values of each period
        for t in range(par.T):
            
            if t == 0:
               kkappalag = 1.0 # set initial demand shock
            else:
               kkappalag = kkappa[t-1] # set demand shock from previous period
               
            kkappa[t] = np.exp(par.rho * np.log(kkappalag) + shock_series[t]) # AR(1) process of demand shocks
            ll[t] = (((1-par.eta)*kkappa[t])/par.w)**(1/par.eta) # use policiy from question 1
            profit[t] = kkappa[t]*(ll[t]**(1-par.eta))-par.w*ll[t] # calculate profit

            if t == 0:
                lllag = 0.0 # set initial number of employees
            else: 
                lllag = ll[t-1] # set number of employees from previous period
            
            adjustment_cost = 0.0 if ll[t] == lllag else par.iota # calculate adjustment cost only if number of employees changes
            discounted_profit = profit[t] - adjustment_cost
            discounted_value = discounted_profit * par.R**(-t) # calculate h in t
            salon_value += discounted_value # fill salon_value each period
            
        return salon_value

    def calc_k2(self, shock_series): 
        """ calculate ex post value of the salon with new policy
        
        Args: 
            self: instance of the class
            shock series: random series of shocks
        
        Returns:
            h: ex post value of the salon
        
        """   
        par = self.par
        
        kkappa = np.zeros(par.T)
        ll = np.zeros(par.T)
        profit = np.zeros(par.T)
        
        salon_value = 0.0
        
        for t in range(par.T):
            
            if t == 0:
               kkappalag = 1.0 
            else:
               kkappalag = kkappa[t-1]
               
            kkappa[t] = np.exp(par.rho * np.log(kkappalag) + shock_series[t])
        

            if t == 0:
                lllag = 0.0
            else: 
                lllag = ll[t-1]

            ll[t] = (((1-par.eta)*kkappa[t])/par.w)**(1/par.eta)
            
            # update policy only if it is more distant than delta to the optimal one
            if abs(lllag - ll[t]) > par.delta:
                ll[t] = ll[t]
            else: 
                ll[t] = lllag
            
            profit[t] = kkappa[t]*(ll[t]**(1-par.eta))-par.w*ll[t]
            
            adjustment_cost = 0.0 if ll[t] == lllag else par.iota
            discounted_profit = profit[t] - adjustment_cost
            discounted_value = discounted_profit * par.R**(-t)
            salon_value += discounted_value
            
        return salon_value   

    def calc_k3(self, shock_series, l3): 

        """ calculate ex post value of the salon with new policy updating delta
        
        Args: 
            self: instance of the class
            shock series: random series of shocks
            l3: delta maximising
        
        Returns:
            h: ex post value of the salon
        
        """  
        par = self.par
        
        # update delta
        par.delta = l3
   
        kkappa = np.zeros(par.T)
        ll = np.zeros(par.T)
        profit = np.zeros(par.T)
   
        salon_value = 0.0
        
        for t in range(par.T):
            
            if t == 0:
               kkappalag = 1.0 
            else:
               kkappalag = kkappa[t-1]
               
            kkappa[t] = np.exp(par.rho * np.log(kkappalag) + shock_series[t])

            if t == 0:
                lllag = 0.0
            else: 
                lllag = ll[t-1]

            ll[t] = (((1-par.eta)*kkappa[t])/par.w)**(1/par.eta)

            if abs(lllag - ll[t]) > par.delta:
                ll[t] = ll[t]
            else: 
                ll[t] = lllag
            
            profit[t] = kkappa[t]*(ll[t]**(1-par.eta))-par.w*ll[t]
            
            adjustment_cost = 0.0 if ll[t] == lllag else par.iota
            discounted_profit = profit[t] - adjustment_cost
            discounted_value = discounted_profit * par.R**(-t)
            salon_value += discounted_value
            
        return salon_value  

    def func_kk(self,l3):
        """ calculate ex ante expected value of the salon """

        par = self.par
        simulations = range(100)
        np.random.seed(0)

        salon_values = []

        for k in simulations: 
            shock_series = np.random.normal(-0.5 * (par.sigmae**2), par.sigmae, par.T)
            salon_value = self.calc_k3(shock_series,l3)
            salon_values.append(salon_value)
        
        expected_value = np.mean(salon_values) 

        return expected_value
        
    def optimal_delta(self):
        """ calculate optimal delta """
        par = self.par
    
        obj4= lambda l3: -self.func_kk(l3)
        res = optimize.minimize_scalar(obj4,method='bounded',bounds=(0.000001, 0.99))
        l3 = res.x
        
        return l3
    

    def calc_k4(self, shock_series): 
        """ calculate ex post value of the salon with our new policy """
        par = self.par
        
        kkappa = np.zeros(par.T)
        ll = np.zeros(par.T)
        profit = np.zeros(par.T)
        ideal_profit = np.zeros(par.T)
        std_profit = np.zeros(par.T)
        salon_value = 0.0
        
        for t in range(par.T):
            
            if t == 0:
               kkappalag = 1.0 
            else:
               kkappalag = kkappa[t-1]
               
            kkappa[t] = np.exp(par.rho * np.log(kkappalag) + shock_series[t])
        
            if t == 0:
                lllag = 0.0
            else: 
                lllag = ll[t-1]

            ll[t] = (((1-par.eta)*kkappa[t])/par.w)**(1/par.eta)

            std_profit[t] = kkappa[t]*(lllag**(1-par.eta))-par.w*lllag # profits with previous policy
            ideal_profit[t] = kkappa[t]*(ll[t]**(1-par.eta))-par.w*ll[t] # profits with optimal policy
            if ideal_profit[t] - std_profit[t] > par.iota : # if the difference is larger than iota, update policy
               ll[t] = ll[t] 
            else:
                ll[t] = lllag
               
            profit[t] = kkappa[t]*(ll[t]**(1-par.eta))-par.w*ll[t]
            adjustment_cost = 0.0 if ll[t] == lllag else par.iota
            discounted_profit = profit[t] - adjustment_cost
            discounted_value = discounted_profit * par.R**(-t)
            salon_value += discounted_value
            
        return salon_value 
        