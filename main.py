import mysql.connector
import pandas as pd
import numpy as np
import pickle
import sys
import random
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import TemporalAbstraction
import RTPmining
import classifier
import Pattern_sequence

def store_patterns(i,trainC1,trainC0,g,C1_support,C0_support,alignment):	
	support=C1_support*len(trainC1)
	C1Patterns = RTPmining.pattern_mining(trainC1, g, support)
	
	support=C0_support*len(trainC0)
	C0Patterns = RTPmining.pattern_mining(trainC0, g, support)
	
	############### Writing patterns to the files and pickle #################
	C1Patterns_file = open('C1_patterns_'+alignment+'_fold'+str(i)+'.txt', 'w')
	C0Patterns_file = open('C0_patterns_'+alignment+'_fold'+str(i)+'.txt', 'w')
	for p in C1Patterns:
		C1Patterns_file.write(p.describe())
	for p in C0Patterns:
		C0Patterns_file.write(p.describe())
	f1_name = 'C1Patterns_'+alignment+'_fold'+str(i)+'.pckl'
	f0_name = 'C0Patterns_'+alignment+'_fold'+str(i)+'.pckl'
	f = open(f1_name, 'wb')
	pickle.dump(C1Patterns, f)
	f.close()
	f = open(f0_name, 'wb')
	pickle.dump(C0Patterns, f)
	f.close()

	return C1Patterns, C0Patterns

def load_patterns(i, alignment):

	f1_name = 'C1Patterns_'+alignment+'_fold'+str(i)+'.pckl'
	f0_name = 'C0Patterns_'+alignment+'_fold'+str(i)+'.pckl'

	f = open(f1_name, 'rb')
	C1Patterns = pickle.load(f)
	f.close()
	f = open(f0_name, 'rb')
	C0Patterns = pickle.load(f)
	f.close()

	return C1Patterns, C0Patterns

def random_subset( iterator, K ):
	result = []
	N = 0
	for item in iterator:
		N += 1
		if len( result ) < K:
			result.append( item )
		else:
			s = int(random.random() * N)
			if s < K:
				result[ s ] = item
	return result

def pre_process(data):
	selected = data.CurrentLocationTypeCode == 'ICU   '
	data.loc[selected, 'CurrentLocationTypeCode'] = 'ICU'
	selected = data.CurrentLocationTypeCode == 'NURSE '
	data.loc[selected, 'CurrentLocationTypeCode'] = 'NURSE'

	data.OrganFailure = data.OrganFailure.astype(float)
	data.InflammationFlag = data.InflammationFlag.astype(float)
	data.InfectionFlag = data.InfectionFlag.astype(float)
	data.SystolicBP = data.SystolicBP.astype(float)
	data.DiastolicBP = data.DiastolicBP.astype(float)
	data.HeartRate = data.HeartRate.astype(float)
	data.RespiratoryRate = data.RespiratoryRate.astype(float)
	data.Temperature = data.Temperature.astype(float)
	data.PulseOx = data.PulseOx.astype(float)
	data.FIO2 = data.FIO2.astype(float)
	data.OxygenFlow = data.OxygenFlow.astype(float)
	data.BUN = data.BUN.astype(float)
	data.Procalcitonin = data.Procalcitonin.astype(float)
	data.WBC = data.WBC.astype(float)
	data.Bands = data.Bands.astype(float)
	data.Lactate = data.Lactate.astype(float)
	data.Platelet = data.Platelet.astype(float)
	data.Creatinine = data.Creatinine.astype(float)
	data.MAP = data.MAP.astype(float)
	data.BiliRubin = data.BiliRubin.astype(float)
	data.CReactiveProtein = data.CReactiveProtein.astype(float)
	data.SedRate = data.SedRate.astype(float)

	selected = data.MinutesFromArrival == 0
	data.loc[selected, 'InflammationFlag'] = np.nan

	return data

def cut_nonshock_seq(shock_event, nonshock_event):
	shock_lengths = shock_event.groupby(['VisitIdentifier']).MinutesFromArrival.count().tolist()
	shock_lengths.sort()
	new_nonshock_event = pd.DataFrame()
	grouped = nonshock_event.groupby(['VisitIdentifier'])
	i = 0
	for name, group in grouped:
		group = group.sort_values(['MinutesFromArrival'])
		



def make_MSS(shock_event, nonshock_event, min, alignment, settings):
	# selected = (shock_event.ShockFlag != 1)
	# shock_event = shock_event.loc[selected,:]

	if alignment == 'right':
		# shock_cut = shock_event[shock_event.ShockTime - shock_event.MinutesFromArrival >= min]
		shock_cut = shock_event[shock_event.LastMinute - shock_event.MinutesFromArrival >= min]
		nonshock_cut = nonshock_event[nonshock_event.LastMinute - nonshock_event.MinutesFromArrival >= min]

	elif alignment == 'left':
		shock_cut = shock_event[shock_event.MinutesFromArrival <= min]
		nonshock_cut = nonshock_event[nonshock_event.MinutesFromArrival <= min]
			
	if settings == 'trunc':
		shock_event = shock_cut
		nonshock_event = nonshock_cut

	# balance the +/- samples
	if len(nonshock_event.VisitIdentifier.unique()) > len(shock_event.VisitIdentifier.unique()):
		nonshock_id = random_subset(nonshock_event.VisitIdentifier.unique(), len(shock_event.VisitIdentifier.unique()))
		nonshock_event = nonshock_event[nonshock_event.VisitIdentifier.isin(nonshock_id)]
	elif len(nonshock_event.VisitIdentifier.unique()) < len(shock_event.VisitIdentifier.unique()):
		shock_id = random_subset(shock_event.VisitIdentifier.unique(), len(nonshock_event.VisitIdentifier.unique()))
		shock_event = shock_event[shock_event.VisitIdentifier.isin(shock_id)]

	for f in TemporalAbstraction.Lab_Features:
		shock_event[f.name], nonshock_event[f.name] = TemporalAbstraction.abstraction_alphabet(shock_event[f.name], nonshock_event[f.name])

	MSS_shock = []
	visitid_shock = []
	grouped = shock_event.groupby('VisitIdentifier')
	for name, group in grouped:
		group = group.sort_values(['MinutesFromArrival'])
		MSS_shock.append(TemporalAbstraction.MultivariateStateSequence(group))
		visitid_shock.append(name)

	MSS_nonshock = []
	visitid_nonshock = []
	grouped = nonshock_event.groupby('VisitIdentifier')
	for name, group in grouped:
		group = group.sort_values(['MinutesFromArrival'])
		MSS_nonshock.append(TemporalAbstraction.MultivariateStateSequence(group))
		visitid_nonshock.append(name)

	f = open('MSS_shock.pckl', 'wb')
	pickle.dump(MSS_shock, f)
	f.close()
	f = open('MSS_nonshock.pckl', 'wb')
	pickle.dump(MSS_nonshock, f)
	f.close()
	f = open('shock_cut.pckl', 'wb')
	pickle.dump(shock_cut, f)
	f.close()
	f = open('nonshock_cut.pckl', 'wb')
	pickle.dump(nonshock_cut, f)
	f.close()
	f = open('visitid_shock.pckl', 'wb')
	pickle.dump(visitid_shock, f)
	f.close()
	f = open('visitid_nonshock.pckl', 'wb')
	pickle.dump(visitid_nonshock, f)
	f.close()

	return MSS_shock, MSS_nonshock, shock_cut, nonshock_cut, visitid_shock, visitid_nonshock

def early_prediction(shock_cut, nonshock_cut, MSS_shock, MSS_nonshock, visitid_shock, visitid_nonshock, g, C1_support, C0_support, settings, alignment):
	print "size of nonshock data:", len(MSS_nonshock)
	print "size of shock data:", len(MSS_shock)

	num_folds = 5
	C0_subset_size = len(MSS_nonshock)/num_folds
	C1_subset_size = len(MSS_shock)/num_folds
	
	test_pred, test_pred_prob, train_pred, test_labels = [], [],[], []
	for i in range(num_folds):
		print "***************** FOLD ", i+1, "*****************"
		# 1. Generate test and train MSS based on settings
		trainC0 = MSS_nonshock[:i*C0_subset_size] + MSS_nonshock[(i+1)*C0_subset_size:]
		trainC0_visitid = visitid_nonshock[:i*C0_subset_size] + visitid_nonshock[(i+1)*C0_subset_size:]
		trainC1 = MSS_shock[:i*C1_subset_size] + MSS_shock[(i+1)*C1_subset_size:]
		trainC1_visitid = visitid_shock[:i*C1_subset_size] + visitid_shock[(i+1)*C1_subset_size:]

		if settings == 'trunc':
			testC0 = MSS_nonshock[i*C0_subset_size:][:C0_subset_size]
			testC0_visitid = visitid_nonshock[i*C0_subset_size:][:C0_subset_size]
			testC1 = MSS_shock[i*C1_subset_size:][:C1_subset_size]
			testC1_visitid = visitid_shock[i*C1_subset_size:][:C1_subset_size]

		elif settings == 'entire':
			testC1_visitid_all = visitid_shock[i*C1_subset_size:][:C1_subset_size]
			test_shock = shock_cut[shock_cut.VisitIdentifier.isin(testC1_visitid_all)]
			testC1_visitid = test_shock.VisitIdentifier.unique()
			testC0_visitid_all = visitid_nonshock[i*C0_subset_size:][:C0_subset_size]
			test_nonshock = nonshock_cut[nonshock_cut.VisitIdentifier.isin(testC0_visitid_all)]
			testC0_visitid = test_nonshock.VisitIdentifier.unique()
			if len(test_nonshock.VisitIdentifier.unique()) > len(test_shock.VisitIdentifier.unique()):
				nonshock_id = random_subset(test_nonshock.VisitIdentifier.unique(), len(test_shock.VisitIdentifier.unique()))
				test_nonshock = test_nonshock[test_nonshock.VisitIdentifier.isin(nonshock_id)]
			elif len(test_nonshock.VisitIdentifier.unique()) < len(test_shock.VisitIdentifier.unique()):
				shock_id = random_subset(test_shock.VisitIdentifier.unique(), len(test_nonshock.VisitIdentifier.unique()))
				test_shock = test_shock[test_shock.VisitIdentifier.isin(shock_id)]
			testC1=[]
			grouped = test_shock.groupby('VisitIdentifier')
			for name, group in grouped:
				group = group.sort_values(['MinutesFromArrival']).reset_index()
				testC1.append(TemporalAbstraction.MultivariateStateSequence(group))
			testC0=[]
			grouped = test_nonshock.groupby('VisitIdentifier')
			for name, group in grouped:
				group = group.sort_values(['MinutesFromArrival']).reset_index()
				testC0.append(TemporalAbstraction.MultivariateStateSequence(group))

		print "Size of shock and nonshock training:", len(trainC1), len(trainC0)
		print "Size of shock test:", len(testC1)
		print "Size of non-shock test:", len(testC0)

		# 2. either generate the patterns or load the dumped ones
		C1Patterns, C0Patterns = store_patterns(i,trainC1,trainC0,g,C1_support,C0_support,alignment)
		# C1Patterns, C0Patterns = load_patterns(i,alignment)
		allPatterns = list(C1Patterns)
		for j in range(0,len(C0Patterns)):
			if not any((x == C0Patterns[j]) for x in allPatterns):
				allPatterns.append(C0Patterns[j])
		print "number of all patterns:", len(allPatterns)

		# 3. Create the binary matrix of train and test sets based on the patterns extracted
		trainData = list(trainC1)
		trainData.extend(trainC0)
		trainLabels = list(np.ones(len(trainC1)))
		trainLabels.extend(np.zeros(len(trainC0)))
		X_train = Pattern_sequence.create_pattern_sequence(trainData, allPatterns)
		y_train = np.vstack(trainLabels)
		print "train:", len(X_train), len(y_train)

		testData = list(testC1)
		testData.extend(testC0)
		testLabels = list(np.ones(len(testC1)))
		testLabels.extend(np.zeros(len(testC0)))
		X_test = Pattern_sequence.create_pattern_sequence(testData, allPatterns)
		y_test = np.vstack(testLabels)
		print "test:", len(X_test), len(y_test)
		
		# 4. learn LSTM classifier and evaluate the prediction
		trp, tsp, tsp_prob = classifier.learn_lstm(X_train, X_test, y_train, y_test)
		print len(tsp)
		test_labels.extend(testLabels)
		test_pred.extend(tsp)
		train_pred.extend(trp)
		for each in tsp_prob:
			test_pred_prob.append(each[0])
		print len(test_pred)

	print metrics.confusion_matrix(test_labels, test_pred)
	accuracy = accuracy_score(test_labels, test_pred)
	precision = precision_score(test_labels, test_pred)
	recall = recall_score(test_labels, test_pred)
	f_measure = f1_score(test_labels, test_pred)
	fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_pred_prob, pos_label=1)
	auc = metrics.auc(fpr, tpr)
	return accuracy, precision, recall, f_measure, auc


def store_event(cursor):
	## SHOCK CLASS
	query = """ SELECT VisitIdentifier,MinutesFromArrival,AbxDrugName,SystolicBP,DiastolicBP,HeartRate,RespiratoryRate,Temperature,PulseOx,MAP,FIO2,OxygenFlow,CurrentLocationTypeCode,Procalcitonin,WBC,Bands,BUN,Lactate,Platelet,Creatinine,BiliRubin,CReactiveProtein,SedRate,InfectionFlag,InflammationFlag,OrganFailure,ShockFlag
			FROM RTP_Shock_Deceased_all 
			ORDER BY VisitIdentifier"""
	cursor.execute(query)
	shock_event = pd.DataFrame(cursor.fetchall(),columns=[i[0] for i in cursor.description])
	# selected = shock_event.ShockFlag == 1.0
	# shock_event.loc[:,'ShockTime'] = shock_event.loc[selected, 'MinutesFromArrival']
	# shock_event.ShockTime = shock_event.groupby('VisitIdentifier')['ShockTime'].bfill()
	shock_event = pre_process(shock_event)
	shock_event.loc[:,'LastMinute'] = shock_event.groupby('VisitIdentifier').tail(1).MinutesFromArrival
	shock_event.LastMinute = shock_event.groupby('VisitIdentifier')['LastMinute'].bfill()


	## NON-SHOCK CLASS
	query = """ SELECT VisitIdentifier,MinutesFromArrival,AbxDrugName,SystolicBP,DiastolicBP,HeartRate,RespiratoryRate,Temperature,MAP,PulseOx,FIO2,OxygenFlow,CurrentLocationTypeCode,Procalcitonin,WBC,Bands,BUN,Lactate,Platelet,Creatinine,BiliRubin,CReactiveProtein,SedRate,InfectionFlag,InflammationFlag,OrganFailure,ShockFlag
				FROM RTP_Shock_Alive_all 
				ORDER BY VisitIdentifier"""
	cursor.execute(query)
	nonshock_event = pd.DataFrame(cursor.fetchall(),columns=[i[0] for i in cursor.description])
	nonshock_event.loc[:,'LastMinute'] = nonshock_event.groupby('VisitIdentifier').tail(1).MinutesFromArrival
	nonshock_event.LastMinute = nonshock_event.groupby('VisitIdentifier')['LastMinute'].bfill()
	nonshock_event = pre_process(nonshock_event)

	f = open('shock_event.pckl', 'wb')
	pickle.dump(shock_event, f)
	f.close()
	f = open('nonshock_event.pckl', 'wb')
	pickle.dump(nonshock_event, f)
	f.close()

	return shock_event, nonshock_event

def load_event():
	f = open('shock_event.pckl', 'rb')
	shock_event = pickle.load(f)
	f.close()
	f = open('nonshock_event.pckl', 'rb')
	nonshock_event = pickle.load(f)
	f.close()

	return shock_event, nonshock_event




