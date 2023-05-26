import numpy as np
from scipy import optimize
from types import SimpleNamespace
import matplotlib.pyplot as plt

class WorkerClass : 

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.alpha = 0.5
        par.k = 1.0
        par.ni = 1/(2*(16**2))
        par.w = 1.0
        par.tau = 0.30
        par.sigma = 1.001
        par.rho = 1.001
        par.epsilon = 1.0
        
    # define functions

    def optimal_labor(self):
        """ calculate optimal labor function from FOC """

        par = self.par

        piece1 = -par.k
        piece2= np.sqrt((par.k**2) + (4.0*par.alpha*((1-par.tau)*par.w)**2)/par.ni)
        piece3= 2*(1-par.tau)*par.w

        return (piece1+piece2)/piece3
    
    def government(self):
        """ calculate government consumption """

        par=self.par

        lstar = self.optimal_labor()

        return par.tau*par.w*lstar
    
    def utility(self):
        """ calculate utility with the first formulation of the utility function """

        par=self.par

        l_star= self.optimal_labor()
        g_star= self.government()
        consumption = par.k + (1-par.tau)*par.w*l_star

        return np.log((consumption**par.alpha) * (g_star**(1-par.alpha)))-par.ni*(l_star**2)/2
    
    def utility2(self, g, l): 
        """ calculate utility with the second formulation of the utility function
        
        Args: 
            self: instance of the class
            g: government consumption
            l: labor
        
        Returns:
            utility: utility from the second formulation of the utility function
        
        """
        par = self.par

        cc = par.alpha*((par.k + (1-par.tau)*par.w*l)**((par.sigma-1)/par.sigma))
        gg = (1-par.alpha)*(g**((par.sigma-1)/par.sigma))
        central = (cc+gg)**(par.sigma/(par.sigma-1))
        ut = ((central**(1-par.rho))-1)/(1-par.rho)
        disut = par.ni*(l**(1+par.epsilon))/(1+par.epsilon)

        return ut-disut
    
    def optimal_lg(self,g):
        """ calculate optimal labor given g
        
        Args: 
            self: instance of the class
            g: government consumption
        
        Returns:
            lg_optimal: optimal labor given g
        
        """    
        par = self.par

        obj= lambda l: -self.utility2(g,l)
        res = optimize.minimize_scalar(obj,method='bounded',bounds=(0.0, 24.0))
        lg_optimal = res.x

        return lg_optimal
    
    def equation_to_solve(self,g):
        """ calculate G function given L*
        
        Args: 
            self: instance of the class
            g: government consumption
        
        Returns:
            G: G function given L*
        
        """    
        par = self.par

        L_star = self.optimal_lg(g)
        
        return g - par.tau * par.w * L_star
