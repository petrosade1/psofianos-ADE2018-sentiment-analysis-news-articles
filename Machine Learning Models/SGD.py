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

#Unigram Transformation

def unigram_trans(data):
        
        vectorizer = CountVectorizer()

        vectorizer = vectorizer.fit(data)

        return vectorizer
     

#Reading data from the file

def get_data(name):
	 
	data = pd.read_csv(name,header=0,encoding = 'UTF-8')

	X = data['text']

	Y = data['polarity']

	return X, Y



#Transformation with tf-idf

def tfidf_trans(data):

	from sklearn.feature_extraction.text import TfidfTransformer
 
	transformer = TfidfTransformer()

	transformer = transformer.fit(data)

	return transformer

#Transformation to Unigrams + Bigrams

def bigram_trans(data):  

	from sklearn.feature_extraction.text import CountVectorizer

	vectorizer = CountVectorizer(ngram_range=(1,2))

	vectorizer = vectorizer.fit(data)

	return vectorizer

def general_comparison_models(X,Y):

        models = []

        models.append(('SGD', SGDClassifier(loss="hinge", penalty="l2",alpha=0.0005,random_state=0, max_iter=100,tol=None)))

        results = []

        names = []
        
        scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

        for name, model in models:

          kfold = model_selection.KFold(n_splits=5, random_state=7)

          predicted = model_selection.cross_val_predict(model, X, Y, cv=kfold)

          names.append(name)

          print(name)

          precision, recall, fscore, support = score(Y, predicted)

          print('Accuracy:  {:.2%}'.format(accuracy_score(Y, predicted)))

          print('precision: {}'.format(precision))

          print('recall: {}'.format(recall))

          print('fscore: {}'.format(fscore))

          print('support: {}'.format(support))
          
        pass




if __name__ == "__main__":

        import time

        #calculates time for each model

        seed=7           #specifies a seed for a specific state of weights
        
        print ("Retrieving the data in the required format")

        pos_data = pd.read_csv("POS_Tagged.csv",encoding = 'UTF-8')  #reads the pos file

        combi = pos_data['text']
        
        [X, Y] = get_data(name=train_path)             #retrieves the data
        
        
        var = input("Do you want to use POS Tagger? (y/n): ")

        print("You entered " + str(var))

        if (var=='y'):
           X=combi


        #Doing k-cross validation BoW and TF-IDF with different characteristics

        #Trying Unigrams as characteristics

        print ("Trying Unigrams as features")

        unvectorizer = unigram_trans(X) 

        X_uni = unvectorizer.transform(X)
        
        general_comparison_models(X_uni,Y)

        #Trying Unigrams+Bigrams as characteristics

        print ("Trying Combination of Unigrams & Bigrams as features")

        unvectorizer = bigram_trans(X) 

        X_uni = unvectorizer.transform(X)

        general_comparison_models(X_uni,Y)        

        #Trying Unigrams with TF-IDF

        print ("Trying Unigrams with TF-IDF as features")

        unvectorizer = unigram_trans(X) 

        X_uni = unvectorizer.transform(X)

        uni_tfidf_transformer = tfidf_trans(X_uni)

        X_uni = uni_tfidf_transformer.transform(X_uni)
       
        general_comparison_models(X_uni,Y)
 
        #Trying Unigrams+Bigrams TF-IDF as characteristics

        print ("Trying Combination of Unigrams & Bigrams with TF-IDF as features")

        unvectorizer = bigram_trans(X) 

        X_uni = unvectorizer.transform(X)

        uni_tfidf_transformer = tfidf_trans(X_uni)

        X_uni = uni_tfidf_transformer.transform(X_uni)

        general_comparison_models(X_uni,Y)

pass
