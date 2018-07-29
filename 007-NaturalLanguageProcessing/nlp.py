# -*- coding: utf-8 -*-

# Natural Language Processing

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the Dataset
# *.csv - Comma-Seperated Values
# *.tsv - Tab-Seperated Values
# Being as some of the reviews contain commas, we use a tsv
# To avoid possible issues with double-quotes, we include 'quoting=3'
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

## - -- --- Cleaning the Text --- -- - ##
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for review in range(0, 1000):
	# Replace everything (punctuation) except letters with a space
	r = re.sub('[^a-zA-Z]',' ', dataset['Review'][review])
	# Make all letters lowercase
	r = r.lower()
	# Remove non-significant words (articles, prepositions, etc)
	# Split into seperate words first
	# Compare words of review to stopwords, removing the stopwords
	# Use word stemming (loved, loving, loves, etc becomes love)
	r = r.split()
	ps = PorterStemmer()
	r = [ps.stem(word) for word in r if not word in set(stopwords.words('english'))]
	# Rejoin review into string format
	r = ' '.join(r)
	corpus.append(r)

## - -- --- Create Bag of Words Model --- -- - ##
# Compile unique words (tokenization)from reviews into a sparse matrix 
# with corresponding reviews and appearance (of word) count. We'll use 
# this sparse matrix of independent variables for our classification.
# (Most of our above cleaning can be done by parameters of the
# CountVectorizer())
# Consider max_features parameter when processing large texts
# The sparse matrix can be further reduced by dimensionality reduction,
# as will be seen in future machine learning models.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# As you'll notice, thanks to our previous work with classification, we merely
# need to copy and paste the actual classification process.
# Typical classification methods used in NLP are NBC, DTC, and RFC. We'll be
# using Naive Bayes Classification.

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Type in 'cm' in console to print confusion matrix
# ([55, 42]  55+91 = 146 (Correct Predictions)
#  [12, 91]) 12+42 = 54 (Incorrect Predictions)


# Feel Free To Play With Other Classification Models In Attempts To Better
# The Overall Accuracy On Such A Small DataSet.
