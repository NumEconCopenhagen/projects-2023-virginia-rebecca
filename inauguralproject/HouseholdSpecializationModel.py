
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.deltalm = 0.5
        par.deltalf = 0.5
        par.deltahm = 0.9
        par.deltahf = 1
  

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # e.1 addition to the model
        

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
           H = HM**(1-par.alpha)*HF**par.alpha
           

        elif par.sigma == 0:
           H = np.minimum(HM, HF)
           

        else:
          H = ((1-par.alpha)*HM**((par.sigma -1 )/par.sigma) + par.alpha*HF**((par.sigma -1 )/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        par=self.par
        sol=self.sol
        optimum=SimpleNamespace()

        # Define objective function of the minimization, therefore the utility function with the minus in front
        obj= lambda x: -self.calc_utility(x[0], x[1], x[2],x[3])
        
        # Set the initial guess
        initial_guess = [4, 4, 4, 4]

        # Set the bounds
        bounds=((1e-8,24), (1e-8,24), (1e-8,24), (1e-8,24))

        # Apply the minimize function
        sol_cont = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds)
        
        # Save the results as LM, HM, LF, HF
        optimum.LM=sol_cont.x[0]
        optimum.HM=sol_cont.x[1]
        optimum.LF=sol_cont.x[2]
        optimum.HF=sol_cont.x[3]

        return optimum   

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par=self.par
        sol=self.sol
        max=SimpleNamespace()

        # Loop trough the index and the values of wF_vec
        for i,wF in enumerate(par.wF_vec):
           
           # Set the parameter value as the correspondent value of the loop
           par.wF = wF

           # Define objective function of the minimization, therefore the utility function with the minus in front
           obj= lambda x: -self.calc_utility(x[0], x[1], x[2],x[3])
        
           # Set the initial guess
           initial_guess = [4, 4, 4, 4]

           # Set the bounds
           bounds=((1e-8,24), (1e-8,24), (1e-8,24), (1e-8,24))

           # Apply the minimize function
           solution = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds)

           # Take results of HM and HF
           max.HM=solution.x[1]
           max.HF=solution.x[3]
           
           # Stack the solutions of each loop in the vectors HM_vec and HF_vec
           sol.HM_vec[i]=solution.x[1]
           sol.HF_vec[i]=solution.x[3]
        

        return sol.HM_vec, sol.HF_vec

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def objective(self, x):
        """ calculate the objective function"""

        par = self.par
        sol = self.sol
        
        # Update the parameters for alpha and sigma
        par.alpha = x[0]
        par.sigma = x[1]
        
        # Calculate the optimal HM and HF vectors
        self.solve_wF_vec()
        
        # Run the regression
        self.run_regression()
        
        # Calculate the function that has to be minimized
        objective = (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        return objective
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol
        alphasigma=SimpleNamespace()
        
    
        # Set initial guess
        x0 = [0.5, 1]

        # Set bounds for alpha and sigma
        bounds = ((0.0,1.),(0.0,None))
        
        # Apply the minimization
        res = optimize.minimize(self.objective,x0,bounds=bounds,method='Nelder-Mead')
        
        # Save the results
        alphasigma.alpha= res.x[0]
        alphasigma.sigma = res.x[1]
        alphasigma.fun = res.fun
        
        #Print the results
        print(f'optimal alpha = {alphasigma.alpha:.4f}')
        print(f'optimal sigma = {alphasigma.sigma:.4f}')
        print(f'optimal function = {alphasigma.fun:.4f}')

        return res
    
    def calc_utility_addition(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
           H = HM**(1-par.alpha)*HF**par.alpha
           

        elif par.sigma == 0:
           H = np.minimum(HM, HF)
           

        else:
          H = ((1-par.alpha)*HM**((par.sigma -1 )/par.sigma) + par.alpha*HF**((par.sigma -1 )/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_) 
        
        return utility - disutility - par.deltahm*(HM**2)- par.deltahf*(HF**2) - par.deltalm*(LM**2) - par.deltalf*(LF**2)
    
    def solve_wF_vec_addition(self,discrete=False):
        """ solve model for vector of female wages """

        par=self.par
        sol=self.sol
        max=SimpleNamespace()

        # Loop trough the index and the values of wF_vec
        for i,wF in enumerate(par.wF_vec):
           
           # Set the parameter value as the correspondent value of the loop
           par.wF = wF

           # Define objective function of the minimization, therefore the utility function with the minus in front
           obj= lambda x: -self.calc_utility_addition(x[0], x[1], x[2],x[3])
        
           # Set the initial guess
           initial_guess = [4, 4, 4, 4]

           # Set the bounds
           bounds=((1e-8,24), (1e-8,24), (1e-8,24), (1e-8,24))

           # Apply the minimize function
           solution = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds)

           # Take results of HM and HF
           max.HM=solution.x[1]
           max.HF=solution.x[3]
           
           # Stack the solutions of each loop in the vectors HM_vec and HF_vec
           sol.HM_vec[i]=solution.x[1]
           sol.HF_vec[i]=solution.x[3]
        

        return sol.HM_vec, sol.HF_vec
    
    def objective_addition(self, x):
        """ calculate the objective function"""

        par = self.par
        sol = self.sol
        
        # Update the parameter sigma
        par.sigma = x[0]
        par.deltahm = x[1]
        par.deltahf = x[2]
        par.deltalm = x[3]
        par.deltalf = x[4]
        
        # Calculate the optimal HM and HF vectors
        solver = self.solve_wF_vec_addition()
        
        # Run the regression
        regression = self.run_regression()
        
        # Calculate the function that has to be minimized
        objective = (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        return objective
    
    def estimate_addition(self,alpha=None,sigma=None):
        """ sigma """

        par = self.par
        sol = self.sol
        alphasigma=SimpleNamespace()
        
        # Set initial guess
        x0 = (1, 0.5, 0.5, 0.5, 0.5)

        # Set bounds for alpha and sigma
        bounds = ((0.0,None), (None, None), (None, None), (None, None), (None, None))
        
        # Apply the minimization
        res = optimize.minimize(self.objective_addition,x0,method='Nelder-Mead', bounds=bounds)
        
        # Save the results
        alphasigma.sigma = res.x[0]
        alphasigma.deltahm = res.x[1]
        alphasigma.deltahf = res.x[2]
        alphasigma.deltalm = res.x[3]
        alphasigma.deltalf = res.x[4]
        alphasigma.fun = res.fun
        
        #Print the results
        print(f'optimal sigma = {alphasigma.sigma:.4f}')
        print(f'optimal deltahm = {alphasigma.deltahm:.4f}')
        print(f'optimal deltahf = {alphasigma.deltahf:.4f}')
        print(f'optimal deltalm = {alphasigma.deltalm:.4f}')
        print(f'optimal deltalf = {alphasigma.deltalf:.4f}')
        print(f'optimal function = {alphasigma.fun:.4f}')

        return res

