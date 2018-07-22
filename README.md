# Machine Learning
> _"Field of study that gives computers the ability to learn without being explicitly programmed"_ \- Arther Samuel

Code is wittern in python using scikit-learn library. The Artificial Neural Networks and Convolutional 
Neural Networks code is written using the keras library which is based on tensorflow.

# Terminology
- **Feature Vector** is the set of values of the depedent variable. eg.   
(age, work experience, gender) = (21, 2, "Male") is a feature vector.
- **Value** is the independent variable associate with each feature vector eg.  
(age = 21, work exp = 2, gender = "Male") => (salary = 25000)
- **Feature Matrix** it is a set of feature vectors arranged in a matrix.
- **Training Set** : Using a set vector-value pairs we "train" our machine to 
make predictions. This set is called the training set  
_Note : the word "train" means to optimize weights that are used to make predictions_
- **Test Set** : Once our model is trained, we would like to test its efficieny
to make correct predictions we do this by using the vectors from the test-set, allow 
our machine to make a prediction, and then "compare" our pridiction with the actual value
correcponding to this vector.
_Note : the word "compare" is  sometimes mathematically defined_

# Classification
1. **Supervised learning** : start with a feature vector and value pair and goal is to predict value
for a previously unseen feature vector.  
1.1 **Regression Model** : predicted value is a real number.  
1.2 **Classification**   : predicted value is a label. (form a finite set)  
Sometimes for aiding visualization and plotting, we only chose those components from the feature vector
that have maximum impact on the decision variables. This choice is based on mathematical interpretation 
of "maximum impact" (R-value and Adjusted-R-value in case of regression and PCA and LDA in case of classification)

2. **Unsupervised learning** : start with feature vector only (no values)  
2.1 **Clustering** : Define some metric to figure out "how similiar" two feature vectors are. 
If they are "similar enough", group them together. Goal is to find these groupings.

3. **Renforcement learning** : We don't start with any feature vectors at all. Data is processed as it is generated.
If a particular feature vector tend to increase (or decrease) the value we are interested in, the machine tends to 
favour that feature vector. this "favouring" is done by the machine being "rewarded" for predicting correctly and 
"penalized" for incorrect predictions.  
3.1 Upper Confidence Bound (UCB) : Deterministic; requires update every round.  
3.2 Thompson Sampling (TS) : Probabilistic; accomodates delayed feedback.  
_NOTE : Supervised learing is about "exploring" possible correlations. Once we have establised a realtion 
we then "exploit" it to our benefit. Renforcement learing is about "exploring" and "exploting" simultaneously. 
Machine "explores" as data is generated and in the very next "round" tries to "exploit" what it has learned till now._ 

# Improving Efficiency

- **Feature Selection** : Consider only those features that contribute to decide dependent variable class or value. 
The fewer the number of variables aids in visualisation. (For regression we do feature selection)  
  1. Forward Selection  
  2. Backward Elimination

- **Feature Extraction** : From existing feature variable, create new feature variables that are fewer in number for
aiding visualising. (For classifiction we de feature extraction).  
  1. Principle Componet Analysis  : Unsupervised. Based on maximising varience.  
  2. Linear Discriminant Analysis : Supervised. Based on maximising class seperation.  
  3. Kernal PCA : For non-linearly seperable classification problems.


 

