import mysql.connector
import pandas as pd
import numpy as np
import pickle
import sys
import random
import math
import multiprocessing
from multiprocessing import Pool
from functools import partial
from sklearn import metrics

import TemporalAbstraction
import RTPmining
import classifier
import main

cnx = mysql.connector.connect('xxxx')
cursor = cnx.cursor()

alignment = 'right'
settings = 'trunc'

def load_MSS():
	f = open('MSS_shock.pckl', 'rb')
	MSS_shock = pickle.load(f)
	f.close()
	f = open('shock_cut.pckl', 'rb')
	shock_cut = pickle.load(f)
	f.close()
	f = open('nonshock_cut.pckl', 'rb')
	nonshock_cut = pickle.load(f)
	f.close()
	f = open('MSS_nonshock.pckl', 'rb')
	MSS_nonshock = pickle.load(f)
	f.close()
	f = open('visitid_shock.pckl', 'rb')
	visitid_shock = pickle.load(f)
	f.close()
	f = open('visitid_nonshock.pckl', 'rb')
	visitid_nonshock = pickle.load(f)
	f.close()
	return MSS_shock, MSS_nonshock, shock_cut, nonshock_cut, visitid_shock, visitid_nonshock

def parallel_early_prediction(h, shock_event, nonshock_event, alignment, settings):
	j=0
	g_gap = 24*60
	C1_support = 0.16
	C0_support = 0.18
	performances = pd.DataFrame(columns=['Hours', 'Accuracy','Precision', 'Recall', 'F-measure', 'AUC'])

	MSS_shock, MSS_nonshock, shock_cut, nonshock_cut, visitid_shock, visitid_nonshock = main.make_MSS(shock_event, nonshock_event, h*60, alignment, settings)
	accuracy, precision, recall, f_measure, auc = main.early_prediction(shock_cut, nonshock_cut, MSS_shock, MSS_nonshock, visitid_shock, visitid_nonshock, g_gap, C1_support, C0_support, settings, alignment)

	performances.loc[j] = [h, accuracy, precision, recall, f_measure, auc]
	print performances
	with open('performances_'+alignment+'_'+settings+'.csv', 'a') as f:
	    performances.to_csv(f, sep='\t', header=False, index=False)


if alignment == 'right':
	hours = np.append(np.arange(0,4,0.5), np.append(range(4,12,1), range(12,31,2)))
	# hours = range(0,50,2)
if alignment == 'left':
	hours = range(4,50,2)

shock_event, nonshock_event = main.store_event(cursor)

pool = Pool(multiprocessing.cpu_count())
parallel_early_prediction_h=partial(parallel_early_prediction, shock_event = shock_event, nonshock_event = nonshock_event, alignment = alignment, settings = settings) 
pool.map(parallel_early_prediction_h, hours)
pool.close()
pool.join()
