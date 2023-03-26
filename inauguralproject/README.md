# Inaugural project

The **results** of the project can be seen from running [inauguralproject.ipynb](inauguralproject.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires no further packages.

The ipynb file import the HouseholdSpecializationModel class, where parameters and methods to solve the project are defined. 

To address the first point, we use a nested for loop for the values of alpha and sigma and we apply a method to solve the model discretely. Then we construct a DataFrame to visualize the results in a table. 

In the second point, we loop through the values of wF and we use the method to solve the model discretely to obtain the two ratios. Then we do a scatter plot to visualize the results. 

In the third point, we follow the same steps of point 2 but we construct a method to solve the model continously that uses a solver. 

To address the fourth point, we construct a method that solves the model for the wF vector. Then we write a method for the objective function that updates the parameters sigma and alpha. Lastly, we construct a method that insert the onjective funtion in a solver so that it finds the optimal alpha and sigma that minimize the objective function. 
