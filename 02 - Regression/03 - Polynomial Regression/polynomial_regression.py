# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Since data set is small we don't split data set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3) # Change degree to see better results
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.figure("Simple Linear Regression")
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Truth or Bluff (Liner Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
plt.figure("Polynomial Linear Regression")
X_grid = np.arange(min(X), max(X), 0.1)  # smoothen graph
X_grid = X_grid.reshape((len(X_grid),1)) # convert vector to matrix
plt.scatter(X,y, color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

print("Salary prediction for a person with 6.5 years of work experience")
# Predicting a new result with Liner Regression
print("Simple linear regression predicts {}".format(lin_reg.predict(6.5)[0]))

# Predicting a new result with Polynomial Regression
print("Polynomial linear regression predicts {}".format(lin_reg_2.predict(poly_reg.fit_transform(6.5))[0]))