Sentiment Analysis on Greek News Articles using Machine Learning and Deep Learning Models Thesis
====================

## Description

   This implementaton examines the problem of sentiment analysis in text, which is the automatic classification of a text document into the sentiment categories positive, negative or neutral according to the authors intention. The dataset that has been used is a collection of news articles in Greek language, that has been extracted through the News API and rated among 2 individuals. The intersection between the ratings sentiment of the 2 raters has been used as initial dataset. The initial dataset has been gone through a variety of different processes such as preprocessing, use of automatic text summarization, use of POS Tagging and the removal in some cases of the neutral category, in order to produce 5 different datasets in total. Then, both a set of different Machine Learning and Deep Learning models were applied on all the datasets, in order to classify the samples of news articles. At the end, as an evaluation method for the models has been used 5-Cross Validation.

 
## Libraries Used

   For the implementation of Machine Learning and Deep Learning Models, have been used mostly the libraries Keras, Scikit Learn, Pandas, Numpy and Tensorflow in conjuction with libraries for the implementation of secondary functions.


## Datasets and Descriptions

   The files used are located in the Datasets Used folder and are each one of them in the Machine Learning and Deep Learning folders, where in the Deep Learning folder are found by changing the suffix of their name to end with 2 (eg Nopreprocess2.csv). Each one of them is explained below:


1. POS_Tagged.csv : Dataset with three categories of sentiment where data pre-processing and POS Tagging has been used.

2. with_neutr_summ_25.csv : Dataset with three categories of sentiment where data pre-processing and automatic text summarization with TextRank has been used.

3. Nopreprocess.csv :  Dataset with three categories of sentiment with no data pre-processing.

4. Preprocessed.csv : Dataset with three categories of sentiment where data pre-processing has been used.

5. Preprocessed_Without_Neutral.csv : Dataset with three categories of sentiment where data pre-processing has been used and also removal of the neutral sentiment category.


   Additionally, in the folder named Not_used_Datasets, there are the data sets that were not used to export the results and have been used to test the different summarization algorithms :

1. With_neutr_summ_25luhn.csv : Dataset with three categories of sentiment where data pre-processing and automatic text summarization with Luhn has been used. 

2. With_neutr_summ_25lsa.csv : Dataset with three categories of sentiment where data pre-processing and automatic text summarization with LSA has been used.

## Machine Learning Models

   In the machine learning folder there are the Machine learning models that have been implemented and the implementation for selecting the parameters of each model. More Specifically:

* LR.py : Implementation of Logistic Regression model and evaluation with Cross Validation.
* NB.py : Implementation of Naive Bayes model and evaluation with Cross Validation.
* RF.py : Implementation of Random Forest model and evaluation with Cross Validation.
* SGD.py : Implementation of Linear SVM with SGD model and evaluation with Cross Validation.
* Train-TestSplit-Models.py : Implementation of all machine learning models for all combinations of features by spliting 80% as training set and 20% as testing set. Evaluation with Cross Validation
* tunerLR.py : Implementation of GridSearch with Cross Validation in order to select the hyperparameters for the Logistic Regression model.
* tunerNB.py : Implementation of GridSearch with Cross Validation in order to select the hyperparameters for the Naive Bayes model.
* tunerSGD.py : Implementation of GridSearch with Cross Validation in order to select the hyperparameters for the Linear SVM with SGD model.

## Deep Learning Models

* LSTM2.py : Implementation of the LSTM (Long Short Term Memory) model that recieves as input all datasets that contain 2 categories of sentiment (ie those that dont include the neutral sentiment category).
* LSTM-CNN2.py : A combination of LSTM (Long Short Term Memory) and CNN (Convolutional Neural Network) models, that recieves as input all datasets that contain 2 categories of sentiment (ie those that dont include the neutral sentiment category).
* LSTM3.py : Implementation of the LSTM (Long Short Term Memory) model that recieves as input all datasets that contain all 3 categories of sentiment.
* LSTM-CNN3.py : A combination of LSTM (Long Short Term Memory) and CNN (Convolutional Neural Network) models, that recieves as input all datasets that contain all 3 categories of sentiment.

## Data Preprocessing

   In the Preprocessing folder there are all the files used to pre-process the data and create all the data sets.

## How to Run
```bash
Examples: 

-python LSTM2.py filename.csv

-python LSTM-CNN3.py filename.csv

-python SGD.py filename.csv
```

## Communication

Author: Petros Sofianos
Email: psofia01@cs.ucy.ac.cy

