from __future__ import print_function

import tensorflow as tf
import sys
import os
import statistics
import numpy as np
import h5py
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.models import model_from_json
from keras.models import Sequential
from argparse import ArgumentParser
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from sklearn.feature_selection import SelectPercentile, f_classif
from keras import metrics
from ast import literal_eval
from keras.layers import Dense, Embedding, Activation, Dropout, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from numpy.random import seed

def accuracy(Ytrain, Ytest):
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n


seed(8)

print('Loading data...')

data = pd.read_csv(sys.argv[1],header=0,encoding = 'UTF-8')
X = data['text']
Y = data['polarity']


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)  #split train/test data

batch_size = 64
epochs=6
max_len = 350
max_words=5000

tokenizer = Tokenizer(max_words)
tokenizer.fit_on_texts(x_train)

x_train= tokenizer.texts_to_sequences(x_train)
x_test= tokenizer.texts_to_sequences(x_test)
x_train=np.array(x_train)
x_test=np.array(x_test)


x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(max_words, embedding_vecor_length,input_length=max_len))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

filepath="w.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#saves best model only
history = model.fit(x_train, y_train, validation_split=0.1 ,epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#load the saved model
print("Loading Best Model Overall")
model.load_weights("w.best.hdf5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


prediction = model.predict_classes(np.array(x_test))
prediction= prediction.flatten()


#Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(accuracy(prediction, np.array(y_test)))

