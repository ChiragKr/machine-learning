# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding catagorical data (text to number(s))
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the "DUMMY VARIABLE TRAP"
X = X[:,1:] # no need to do manually. numpy takes care

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)

# Compare acutal y-values with predicted y-values
print('{0:^12} | {1:^12}'.format('test y-value','predicted y-value'))
print("-------------|------------------")
for i in range(0, len(y_pred)):
	print('{0:^12} | {1:^12}'.format(y_test[i], y_pred[i]))

# =============================================================================
'''
Building the optimal model by removing redundent independent variables that do 
not contribute to the dependent varialbe's value, using Backward Elimination.
'''
import statsmodels.formula.api as sm
#X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis=1) 
#adds columnes of 1's to end of X
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1) #fixed!
X_opt = X[:,[0, 1, 2, 3, 4, 5]] # X_opt = optimal matrix but 1st : "ALL IN!"

# OLS = Ordinary Linear Regressor.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # p-value report (index 2 row with largest p-val. Remove 2)

X_opt = X[:,[0, 1, 3, 4, 5]] # UPDATED X_opt 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # p-value report (index 1 row with largest p-val. Remove 1)

X_opt = X[:,[0, 3, 4, 5]] # UPDATED X_opt 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()# p-value report (index 4 row with largest p-val. Remove 4)

X_opt = X[:,[0, 3, 5]] # UPDATED X_opt  
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()# p-value report (index 5 row with largest p-val. Remove 5)

X_opt = X[:,[0, 3]] # UPDATED X_opt
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()# p-value report (all p-val < 0.05. MODEL READY!)
#==============================================================================
# automatic implementation of Backward Elimination with p-values only:
'''
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL) 
'''
#==============================================================================
# Backward Elimination with p-values and Adjusted R Squared:
'''
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''
#==============================================================================