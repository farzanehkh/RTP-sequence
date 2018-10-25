import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score	
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import GridSearchCV

import RTPmining

# Data = list of MSS, Omega = candidate list of lists, g = gap
def create_binary_matrix(Data, Omega, g):
	binaryMatrix = np.zeros((len(Data),len(Omega)))
	for i in range(0,len(Data)):
		for j in range(0,len(Omega)):
				present = RTPmining.recent_temporal_pattern(Data[i],Omega[j],g)
				if(present):
					binaryMatrix[i,j] = 1
				else:
					binaryMatrix[i,j]= 0
	return binaryMatrix

def learn_lstm(X_train, X_test, y_train, y_test):
	train_pred, test_pred, test_pred_prob = [], [], []

	# 1. zero pad and reshape the test and train data based on the max length of sequence
	max_length_train = max([len(each) for each in X_train])
	max_length_test = max([len(each) for each in X_test])
	max_length = max(max_length_train, max_length_test)

	X_train = sequence.pad_sequences(X_train, maxlen=max_length, truncating = 'pre')
	X_test = sequence.pad_sequences(X_test, maxlen=max_length, truncating = 'pre')

	X_train = np.reshape(X_train, (X_train.shape[0], max_length, X_train.shape[2]))
	print "after reshape", X_train.shape
	X_test = np.reshape(X_test, (X_test.shape[0], max_length, X_train.shape[2]))

	# 2. build LSTM model 
	model = Sequential()
	# model.add(LSTM(40, input_dim=X_train.shape[2], dropout_W=0.2, dropout_U=0.2, return_sequences=True, init='glorot_uniform' ))
	model.add(LSTM(10, input_dim=X_train.shape[2]))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	early_stopping = EarlyStopping(monitor='loss', verbose=1, mode='auto')
	# model.fit(X_train, y_train, epochs=10, batch_size=100, callbacks=[early_stopping])

	# 3. hyper-parameter optimization and learning model
	optimizers = ['rmsprop', 'adam']
	init = ['glorot_uniform', 'normal', 'uniform']
	epochs = [10, 20, 5]
	batches = [50, 70, 100]
	param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
	model.fit(X_train, y_train)

	# 4. evaluation of the model
	predict = model.predict_classes(X_train)
	for each in predict:
		train_pred.append(each[0])
	accuracy = accuracy_score(y_train, train_pred)
	print "training accuracy:", accuracy
	predict = model.predict_classes(X_test)
	for each in predict:
		test_pred.append(each[0])
	accuracy = accuracy_score(y_test, test_pred)
	print "test accuracy:", accuracy
	conf_matrix = metrics.confusion_matrix(y_test,test_pred)
	print conf_matrix
	predict_prob = model.predict_proba(X_test)

	return train_pred, test_pred, predict_prob
