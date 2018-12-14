import pandas as pd 
import pandas
from pandas import DataFrame, read_csv
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
import sys
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier 
import numpy as np
train_path = sys.argv[1] 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def tfidf_trans(data):
	from sklearn.feature_extraction.text import TfidfTransformer 
	transformer = TfidfTransformer()
	transformer = transformer.fit(data)
	return transformer

def bigram_trans(data):  
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(ngram_range=(1,2))
	vectorizer = vectorizer.fit(data)
	return vectorizer

def unigram_trans(data):
        
        vectorizer = CountVectorizer()
        vectorizer = vectorizer.fit(data)

        return vectorizer

def get_data(name):
	 
	data = pd.read_csv(name,header=0,encoding = 'UTF-8')
	X = data['text']
	Y = data['polarity']
	return X, Y




if __name__ == "__main__":
 seed=7
 print("Retrieving the data in the required format")
 [X, Y] = get_data(name=sys.argv[1])
 pos_data = pd.read_csv("POS_Tagged",encoding = 'UTF-8')   
 combi = pos_data['text']

 unvectorizer = bigram_trans(X) 
 X_uni = unvectorizer.transform(X)
 #uni_tfidf_transformer = tfidf_trans(X_uni)
 #X_uni = uni_tfidf_transformer.transform(X_uni)
# Split the dataset in two equal parts
 X_train, X_test, y_train, y_test = train_test_split(
    X_uni, Y, test_size=0.2)

 


 tuned_parameters = [{'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] }]
 

 if (True):
    
    print()
    score=['accuracy']
    clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5,scoring = {'Accuracy': make_scorer(accuracy_score)},refit='Accuracy',return_train_score=True)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print(clf.cv_results_['mean_test_Accuracy'])
    

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

pass
