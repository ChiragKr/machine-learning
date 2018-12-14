# Artificial Neural Network

# Installing Theano (Numerical Commputation Library based on numpy syntax)
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow (Numerical Computation)
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras (based on Theano and Tensorflow)
# pip install --upgrade keras

# =============================================================================
# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # getting rid of one dummy variable to avoid "dummy variable trap"

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (mandatory for ANN)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# =============================================================================
# Creating the Artificial Neural Network
import keras
from keras.models import Sequential # Initialize ANN
from keras.layers import Dense # Add different (hidden) layers randomly initializes weights on synapses

# Initializing the ANN
classifier = Sequential() # plays the role of a classifier therefore 'classifier' = 'ann'

# Adding the input layer and the first hidden layer
classifier.add(Dense(input_dim=11, activation="relu", units=6, kernel_initializer="uniform"))

# Adding the second hidden layer 
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Addinf the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN (applying stochastic gradient descent)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# =============================================================================
# Predicting the Test set results
y_pred = classifier.predict(X_test) # predicted probabilies
y_pred = (y_pred > 0.5) # Threshold = 0.5. (since confusion matrix can only by built using predictions and not probabilities)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# =============================================================================