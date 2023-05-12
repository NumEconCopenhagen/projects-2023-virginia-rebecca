import numpy as np
from scipy import optimize
import sympy as sm
from sympy import Function
from types import SimpleNamespace
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class KrugmanModelClass : 

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.sigma = 3 # greater than one to have positive denominator of par.sigmafrac
        par.gamma = -0.1 # lower than zero to have negative sloped elasticity of consumption so that PP curve is upward sloping (since par.sigma > 1)
        par.Np = 1000 # number of grid points 
        par.p_min = 1 
        par.p_max = 100 
        par.beta = 0.5
        par.alpha = 0.5
        par.w = 10
        par.p=1
        par.L=1
        par.c=1
        par.sigmafrac = (par.sigma -1)/par.sigma
        par.t= 0.2

    def utility(self,c):
        """ calculate utility """

        par = self.par
        
        return par.gamma*c + c**((par.sigma - 1)/par.sigma)
    
    def solve_consumer(self):
        """ solve consumer problem """
        
        par = self.par 

        obj = lambda x: -self.utility(x[0])
        x0 = par.p/2
        cons = lambda x: x[0]*par.p -par.w
        constraints = ({'type':'eq','fun':cons})
        result = optimize.minimize(obj,[x0],method='SLSQP', constraints=constraints)
        consumerss= result.x[0]
        return consumerss
    
    def elasticity(self,c): 
        """" define elasticity function """
        
        par = self.par 

        return (par.sigma*(par.gamma*c**(1/par.sigma) + par.sigmafrac))/par.sigmafrac

    def pp(self,c):
        """ define monopolistic price function """
        
        par = self.par

        return (self.elasticity(c)*par.beta*par.w)/(self.elasticity(c)-1)
    
    def profits(self, pr):
        """ define profit function"""

        par = self.par

        return pr*par.L*par.c -par.alpha*par.w -par.beta*par.L*par.c*par.w
    
    def solve_firm2(self):
        """" solve zero profit condition """
        par = self.par 
        obj = lambda x: self.profits(x)
        result = optimize.root_scalar(obj, bracket=[0.001, 9000], method='brentq')

        return result.root
    
    def number(self,c_eq):
        """ find number of firms in equilibrium """
        par = self.par
        N = round(par.L/(par.alpha + par.beta*par.L*c_eq))
        return N
    
    def pp_tax(self,c):
        """ define monopolistic price function with tax """
        
        par = self.par

        return (self.elasticity(c)*par.beta*par.w*(1 + par.t))/(self.elasticity(c)-1)
    
    def profits_tax(self, pr):
        """ define profit function with tax"""

        par = self.par

        return pr*par.L*par.c -par.alpha*par.w -par.beta*(1+par.t)*par.L*par.c*par.w 
    
    def solve_firm2_tax(self):
        """" solve zero profit condition with tax"""
        par = self.par 
        obj = lambda x: self.profits_tax(x)
        result = optimize.root_scalar(obj, bracket=[0.001, 9000], method='brentq')

        return result.root
    
    
    def number_tax(self,c_eq):
        """ find number of firms in equilibrium with tax"""
        par = self.par
        N = round(par.L/(par.alpha + par.beta*(1+par.t)*par.L*c_eq))
        return N