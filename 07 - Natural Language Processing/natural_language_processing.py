# Natural Language Processing

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd
# =============================================================================
# Importing the dataset (using pandas)
# '.tsv' - "Tab Seperated Values" & '.csv' - "Comma Seperated Values"
# Text might contain ',' but not *tab* therefore '.tsv' better
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3) # ignoring (") by (quoting = 3)
# =============================================================================
# Cleaning the texts
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # Def : "collection of text of same type" 
for i in range(1000):
    # Keep letters for a-z and A-Z. Replace all removals with ' '
    review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][i]) 
    # Convert every letter to lowercase
    review = review.lower() 
    # Convert string into a word list to access individual words.
    review = review.split()
    # Remove past, future tense of same word - eg - "loved" (convert to "love")
    ps = PorterStemmer()
    # Remove non-significant words ("the", "this", "in", etc...)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Reconstruct string from list
    review = ' '.join(review)
    corpus.append(review)
# ============================================================================= 
# Creating the Bag Of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # keep only 1500 most frequent words
X = cv.fit_transform(corpus).toarray()  # independent variables
y = dataset.iloc[:,1].values            # dependent variables

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# =============================================================================
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# =============================================================================