from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, MaxPooling1D, Reshape
from scipy.optimize import minimize
#import keras
#from keras.layers import Input, Dense
#from keras.models import Model


#raw input data is 1888 x 12 (cols 1-12)
#raw output is 1888 (col 14) 1 for true 0 for false
raw_input = np.zeros((1888,12))
raw_label = np.zeros(1888)

#this path needs to be changed for your machine
with open('C:\\Users\\Tony\\Desktop\\data\\drugs.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0  #would be 1 if we had title/name row
    for row in csv_reader:
        for i in range(1,12):
            raw_input[line_count, i] = row[i]
        label = int(row[14])
        if(label > 0): #the file has mulitple classes but here we just reduce to binary
            label = 1
        raw_label[line_count] = label
        line_count += 1

#split train and test data
test_data = raw_input[1500:,]
train_data = raw_input[0:1500,]
train_labels = raw_label[0:1500,]
test_labels = raw_label[1500:,]


#dense neural network
def drug_net():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),

        keras.layers.Dense(64, activation=tf.nn.sigmoid),
        keras.layers.Dense(64, activation=tf.nn.relu),
        #keras.layers.Dense(64, activation=tf.nn.sigmoid),
        #keras.layers.Dense(64, activation=tf.nn.relu),
        #keras.layers.Dense(1064, kernel_regularizer=keras.regularizers.l1(0.01)),

        keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ])

    #optimizer = tf.train.RMSPropOptimizer(0.01)
    #optimizer = tf.train.GradientDescentOptimizer(0.0001)
    #optimizer = tf.train.MomentumOptimizer(0.001,0.1)
    #optimizer = tf.train.FtrlOptimizer(.01)
    #optimizer = tf.train.AdadeltaOptimizer(.01)
    optimizer = tf.train.AdamOptimizer(.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

#prints error during traing (helper function)
def eval(epoch):
    [loss, mae] = drugNN.evaluate(test_data, test_labels, verbose=0)
    #print("Mean Abs Error: {:7.2f}".format(mae))
    print("epoch number:",epoch)

#lets us attatch the eval function to our NN (eval is called every 100 epochs)
class PrintDot_home(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: eval(epoch)

#instantiation of our nn
drugNN = drug_net()

#train nn
rez = drugNN.fit(train_data, train_labels, epochs=500, validation_split=0.2, verbose=0, callbacks=[PrintDot_home()])

#predictions on test data
yhat = drugNN.predict(test_data).flatten()

#yhat wont have 0 and 1, but a float between 0 and 1, we'll make >.5 = 1
yhat = np.where(yhat > 0.5, 1, 0)

#now both yhat and test_labels are binary arrays
#we want to see how many times yhat has the same value as test_labels
#if we subtract the two, the diff: {same->0, diff->1} (after abs val)
accuracy = np.abs(yhat - test_labels)
accuracy = accuracy.astype(int)

#we want to count all the zeros and divide by the size to get an acc%
#invert all the values, then sum all the items, divide by size
accuracy = np.where(accuracy == 0, 1, 0)
sum = np.sum(accuracy)
percent = sum / test_labels.size
print("percent accuracy", percent*100)

