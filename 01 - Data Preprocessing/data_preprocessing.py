# Data Preprocessing


# Importing the libraries
import numpy as np  # mathematical tools
import matplotlib.pyplot as plt  # plot charts. Visualize datasets
import pandas as pd  # import and manage datasets


# Importing the dataset
dataset = pd.read_csv('Data.csv')
# dataset.iloc[row(s),col(s)].values
X = dataset.iloc[:,:-1].values  # a matrix
y = dataset.iloc[:,3].values  # a vector


# Taking care of missing data
from sklearn.preprocessing import Imputer
'''
replace missing data by taking average of the remaining column entries
'''
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])  # calculating average
X[:, 1:3] = imputer.transform(X[:, 1:3])  # replace


# Encoding catagorical data (text to number(s))
'''
Categorical Data: Categorical variables represent types of data which
may be divided into groups. Eg. "Country" data contains 3 catagories 
"France", "Spain" and "Germany". "Purchased" data contains 2 catagories
"Yes" and "No". Text can't be included in a math eq. hence the encoding
ENCODING done using the "LabelEncoder" class)
'''
from sklearn.preprocessing import LabelEncoder 
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) # for country column
'''
"Spain" = 2, "Germany" = 1, "France" = 0 ? We must prevent ML algorithm 
from establishing a relative order over catagorical variable. For this 
we do a DUMMY ENCODING. We have 3 cols(number of cols equal to number 
of catagories) and '1' or '0' is assigned if the row falls in the 
catagory or not! (DUMMY ENCODING done using the "OneHotEncoder" class)
'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])  # '0' = country column which we want to hot encode
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()  # Since y is dependent ML model will know its a catagory and there is no order
y = labelencoder_y.fit_transform(y)  # therefore we need not use "OneHotEncoder".


# Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# Feature Scaling (ML algorithm uses euclidean distance. This ensure one variable does not dominates the distance)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # train set= fit + transform
X_test = sc_X.transform(X_test)  # test set = tranform (ONLY)