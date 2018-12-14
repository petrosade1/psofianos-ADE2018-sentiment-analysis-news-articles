import pandas as pd 
import pandas
from pandas import DataFrame, read_csv
import os
from greek_stemmer import GreekStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import coo_matrix, hstack
from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import MultinomialNB
import sys
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier 
import numpy as np
from sklearn.svm import SVC
train_path = sys.argv[1]
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def unigram_trans(data):
        
        vectorizer = CountVectorizer()
        vectorizer = vectorizer.fit(data)

        return vectorizer
     

def get_data(name):
	 
	data = pd.read_csv(name,header=0,encoding = 'UTF-8')
	X = data['text']
	Y = data['polarity']
	return X, Y

def logistic_regression(Xtrain,Ytrain,Xtest):
         
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=250)
        
        clf.fit(Xtrain, Ytrain)
        
        Ytest = clf.predict(Xtest)
        return Ytest

def Random_Forest(Xtrain,Ytrain,Xtest):
        
        
        clf = RandomForestClassifier(n_estimators=250,n_jobs=-1,random_state=0,max_features="auto")
        clf.fit(Xtrain, Ytrain)
        Ytest = clf.predict(Xtest)
   
        return Ytest

def stochastic_descent(Xtrain, Ytrain, Xtest):
        
   

        clf = SGDClassifier(loss="hinge", penalty="l2",alpha=0.0001,random_state=0, max_iter=np.ceil(40),tol=None)
        print ("SGD Fitting")
        clf.fit(Xtrain, Ytrain)
        print ("SGD Predicting")
        Ytest = clf.predict(Xtest)
        return Ytest

def Naive_Bayes(Xtrain, Ytrain, Xtest):
        clf=MultinomialNB()
        print("NB Fitting")
        clf.fit(Xtrain, Ytrain)
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        print("NB Predicting")
        Ytest = clf.predict(Xtest)
        return Ytest

def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n

def othermetrics(y_test,predicted):
        precision, recall, fscore, support = score(y_test, predicted)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        pass

def tfidf_trans(data):
	from sklearn.feature_extraction.text import TfidfTransformer 
	transformer = TfidfTransformer()
	transformer = transformer.fit(data)
	return transformer

def bigram_trans(data):  #sundiasmos unigrams-bigrams, kane kai sunartisi me mono bigrams
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(ngram_range=(1,2))
	vectorizer = vectorizer.fit(data)
	return vectorizer





if __name__ == "__main__":

        

        seed=7
        
        print ("Retrieving the data in the required format")

        pos_data = pd.read_csv("POS_Tagged.csv",encoding = 'UTF-8')   #reads the POS file

        combi = pos_data['text']
        
        [X, Y] = get_data(name=train_path)
        
        var = input("Do you want to use POS Tagger? (y/n): ")
        print("You entered " + str(var))
        if (var=='y'):
           X=combi

        print ("Splitting the data in training/testing set as 80%/20%")

        Xtrain_text, Xtest_text, Ytrain, YTesto = train_test_split(X ,Y,  test_size=0.2)
        
        
       
        print("Process starting now\n")    
 
        print ("Training Set Results\n")
        

        uni_vectorizer = unigram_trans(Xtrain_text)
        print ("Fitting the unigram model")
        Xtrain_uni = uni_vectorizer.transform(Xtrain_text)

      
        Y_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtrain_uni)
        print ("Accuracy for the Unigram Model with stochastic_descent is ", accuracy(Ytrain, Y_uni))
        Y_uni = Random_Forest(Xtrain_uni, Ytrain, Xtrain_uni)
        print ("Accuracy for the Unigram Model with Random_Forest is ", accuracy(Ytrain, Y_uni))
        Y_uni = logistic_regression(Xtrain_uni,Ytrain,Xtrain_uni)
        print ("Accuracy for the Unigram Model with Logistic Regression is ", accuracy(Ytrain, Y_uni))
        Y_uni = Naive_Bayes(Xtrain_uni,Ytrain,Xtrain_uni)
        print ("Accuracy for the Unigram  Model with Naive Bayes is ", accuracy(Ytrain, Y_uni))    

        print ("\n")

        bi_vectorizer = bigram_trans(Xtrain_text)
        print ("Fitting the bigram model")
        Xtrain_bi = bi_vectorizer.transform(Xtrain_text)

        
        Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
        print ("Accuracy for the Bigram Model with stochastic_descent is ", accuracy(Ytrain, Y_bi))
        Y_bi = Random_Forest(Xtrain_bi, Ytrain, Xtrain_bi)
        print ("Accuracy for the Bigram Model with Random_Forest is ", accuracy(Ytrain, Y_bi))
        Y_bi = logistic_regression(Xtrain_bi,Ytrain,Xtrain_bi)
        print ("Accuracy for the Bigram Model with Logistic Regression is ", accuracy(Ytrain, Y_bi))
        Y_bi = Naive_Bayes(Xtrain_uni,Ytrain,Xtrain_uni)
        print ("Accuracy for the Bigram  Model with Naive Bayes is ", accuracy(Ytrain, Y_bi))
        print ("\n")

        uni_tfidf_transformer = tfidf_trans(Xtrain_uni)
        print ("Fitting the tfidf for unigram model")
        Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)

        Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
        print ("Accuracy for the Unigram TFIDF Model with stochastic_descent is ", accuracy(Ytrain, Y_tf_uni))
        Y_tf_uni = Random_Forest(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
        print ("Accuracy for the Unigram TFIDF Model with Random_Forest is ", accuracy(Ytrain, Y_tf_uni))
        Y_tf_uni = logistic_regression(Xtrain_tf_uni,Ytrain,Xtrain_tf_uni)
        print ("Accuracy for the Unigram TFIDF Model with Logistic Regression is ", accuracy(Ytrain, Y_tf_uni))
        Y_tf_uni = Naive_Bayes(Xtrain_tf_uni,Ytrain,Xtrain_tf_uni)
        print ("Accuracy for the Unigram TFIDF  Model with Naive Bayes is ", accuracy(Ytrain, Y_tf_uni))
        print ("\n")


        bi_tfidf_transformer = tfidf_trans(Xtrain_bi)
        print ("Fitting the tfidf for bigram model")
        Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)

     
        Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
        print ("Accuracy for the biUnigram TFIDF Model with stochastic_descent is ", accuracy(Ytrain, Y_tf_bi))
        Y_tf_bi = Random_Forest(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
        print ("Accuracy for the biUnigram TFIDF Model with Random_Forest is ", accuracy(Ytrain, Y_tf_bi))
        Y_tf_bi = logistic_regression(Xtrain_tf_bi,Ytrain,Xtrain_tf_bi)
        print ("Accuracy for the biUnigram TFIDF Model with Logistic Regression is ", accuracy(Ytrain, Y_tf_bi))
        Y_tf_bi = Naive_Bayes(Xtrain_tf_uni,Ytrain,Xtrain_tf_bi)
        print ("Accuracy for the biUnigram TFIDF Model with Naive Bayes is ", accuracy(Ytrain, Y_tf_bi))
        print ("\n")
       



        print ("Testing Set Results\n")
        
        print("Selected Unigrams as Features\n")
        Xtest_uni = uni_vectorizer.transform(Xtest_text)

        #Trying Stochastic Descent
        print ("Applying the stochastic descent")
        Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
        print ("Accuracy for the Unigram Model with stochastic_descent is ", accuracy(YTesto,Ytest_uni))
        
        print()

        #Trying Random Forest
        print ("Applying the Random Forest")
        Ytest_uni = Random_Forest(Xtrain_uni, Ytrain, Xtest_uni)
        print ("Accuracy for the Unigram Model with Random_Forest is ", accuracy(YTesto, Ytest_uni))

        print()

        #Trying Logistic Regression
        print ("Applying the Logistic Regression")
        Ytest_uni = logistic_regression(Xtrain_uni,Ytrain,Xtest_uni)
        print ("Accuracy for the Unigram  Model with Logistic Regression is ", accuracy(YTesto, Ytest_uni))

        print()

        #Tring Naive Bayes
        print ("Applying the Naive Bayes")
        Ytest_uni = Naive_Bayes(Xtrain_uni,Ytrain,Xtest_uni)
        print ("Accuracy for the Unigram  Model with Naive Bayes is ", accuracy(YTesto, Ytest_uni))

        print ("\n")

        print("Selected Bigrams as Features\n")
        
        Xtest_bi = bi_vectorizer.transform(Xtest_text)

        #Trying Stochastic Descent
        print ("Applying the stochastic descent")
        Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
        print ("Accuracy for the Bigram Model with stochastic descent is ", accuracy(YTesto, Ytest_bi))

        #Trying Random Forest
        print ("Applying the Random Forest")
        Ytest_bi = Random_Forest(Xtrain_bi, Ytrain, Xtest_bi)
        print ("Accuracy for the Bigram Model with Random_Forest is ", accuracy(YTesto, Ytest_bi))

        #Trying Logistic Regression
        print ("Applying the Logistic Regression")
        Ytest_bi = logistic_regression(Xtrain_bi,Ytrain,Xtest_bi)
        print ("Accuracy for the Bigram Model with Logistic Regression is ", accuracy(YTesto, Ytest_bi))

        #Tring Naive Bayes
        print ("Applying the Naive Bayes")
        Ytest_bi = Naive_Bayes(Xtrain_bi,Ytrain,Xtest_bi)
        print ("Accuracy for the Bigram  Model with Naive Bayes is ", accuracy(YTesto, Ytest_bi))
        
        print ("\n")

        print("Selected Unigrams with TF-IDF as Features\n")
        Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)

        #Trying Stochastic Descent
        print ("Applying the stochastic descent")
        Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
        print ("Accuracy for the Unigram TF Model with stochastic descent is ", accuracy(YTesto, Ytest_tf_uni))

        #Trying Random Forest
        print ("Applying the Random Forest")
        Ytest_tf_uni = Random_Forest(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
        print ("Accuracy for the Unigram TF Model with Random_Forest is ", accuracy(YTesto, Ytest_tf_uni))


        #Trying Logistic Regression
        print ("Applying the Logistic Regression")
        Ytest_tf_uni = logistic_regression(Xtrain_tf_uni,Ytrain,Xtest_tf_uni)
        print ("Accuracy for the Unigram TF Model with Logistic Regression is ", accuracy(YTesto, Ytest_tf_uni))


        #Tring Naive Bayes
        print ("Applying the Naive Bayes")
        Ytest_tf_uni = Naive_Bayes(Xtrain_tf_uni,Ytrain,Xtest_tf_uni)
        print ("Accuracy for the Unigram  TF Model with Naive Bayes is ", accuracy(YTesto, Ytest_tf_uni))
        

        print ("\n")




        print("Selected Bigrams with TF-IDF as Features\n")
        Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)

        #Trying Stochastic Descent
        print ("Applying the stochastic descent")
        Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
        print ("Accuracy for the Bigram TF Model with stochastic descent is ", accuracy(YTesto, Ytest_tf_bi))


        #Trying Random Forest
        print ("Applying the Random Forest")
        Ytest_tf_bi = Random_Forest(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
        print ("Accuracy for the Bigram TF Model with Random_Forest is ", accuracy(YTesto, Ytest_tf_bi))


        #Trying Logistic Regression
        print ("Applying the Logistic Regression")
        Ytest_tf_bi = logistic_regression(Xtrain_tf_bi,Ytrain,Xtest_tf_bi)
        print ("Accuracy for the Bigram TF Model with Logistic Regression is ", accuracy(YTesto, Ytest_tf_bi))


        #Tring Naive Bayes
        print ("Applying the Naive Bayes")
        Ytest_tf_bi = Naive_Bayes(Xtrain_tf_bi,Ytrain,Xtest_tf_bi)
        print ("Accuracy for the Bigram  TF Model with Naive Bayes is ", accuracy(YTesto, Ytest_tf_bi))
        


        print ("\n")


        
pass
