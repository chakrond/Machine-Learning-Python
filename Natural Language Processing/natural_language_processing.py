# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 03:36:58 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) # ignor all quoting params = 3

#%% Cleaning
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset.values)):
    review  = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review  = review.lower()
    review  = review.split()
    ps      = PorterStemmer()
    selected_stopwords = stopwords.words('english')
    selected_stopwords.remove('not') # remove 'not' from the set of stopwords
    review  = [ps.stem(word) for word in review if not word in set(selected_stopwords)]
    review  = ' '.join(review)
    corpus.append(review)

# c = [1, 8, 9]
# d = [1, 2, 3]
# a = 1
# [a for a in c if not a in d]

#%% Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_1 = CountVectorizer(max_features=1500 )
X            = vectorizer_1.fit_transform(corpus).toarray()
all_words_1  = vectorizer_1.get_feature_names_out()

y            = dataset.iloc[:, -1].values

vectorizer_2 = CountVectorizer(max_features=1500, analyzer='word', ngram_range=(2, 2))
vectorizer_2.fit_transform(corpus).toarray()
all_words_2  = vectorizer_2.get_feature_names_out()

#%% Data preprocessing

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler() # y is already between 0 and 1, no need to scale
# scaled_X_train = sc_X.fit_transform(X_train)
# scaled_X_test = sc_X.fit_transform(X_test)
# scaled_y_train = sc_y.fit_transform(y_train)

#%% Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train_set, y_train_set = X_train, y_train

c0 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
c1 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
c2 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
c3 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
c4 = GaussianNB().fit(X_train_set, y_train_set)
c5 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
c6 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)

# Title
titles = ['Logistic', 'KNeighbors', 'SVC with Linear kernel', 
         'SVC with RBF kernel', 'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest']

n_model = len(titles)
classifiers = [f'c{i}' for i in range(0, n_model)]
classifiers = [eval(clf) for clf in classifiers]

#%% Making confusion matrix   
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

nrows = 3
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))

for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
for i, clf, title in zip(np.arange(0, n_model), classifiers, titles):
    
    fig.add_subplot(axes.flatten()[i])
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=axes.flatten()[i], colorbar=False)
    y_pred = clf.predict(X_test)
    acc_score = round(accuracy_score(y_test, y_pred), 2)
    axes.flatten()[i].title.set_text(f'{title}, Accuracy Score = {acc_score}')
    plt.axis("on")
    plt.tight_layout()