# Inaugural project

The **results** of the project can be seen from running [inauguralproject.ipynb](inauguralproject.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires no further packages.

The ipynb file imports the HouseholdSpecializationModel class, where parameters and methods to solve the project are defined. 

To address the first point, we used a nested for loop for the values of alpha and sigma and we applied a method to solve the model discretely. Then we constructed a DataFrame to visualize the results in a table. 

In the second point, we looped through the values of wF and we used the method to solve the model discretely to obtain the two ratios. Then we did a scatter plot to visualize the results. 

In the third point, we followed the same steps of point 2 but we constructed a method to solve the model continously that uses a solver. 

To address the fourth point, we constructed a method that solves the model for the wF vector. Then we wrote a method for the objective function that updates the parameters sigma and alpha. 
Lastly, we built a method that insert the objective funtion in a solver so that it finds the optimal alpha and sigma that minimize the objective function. 

In the fifth point, we extended the model so it includes satisfaction that (domestic and non domestic) work can bring despite the effort in doing it. 

