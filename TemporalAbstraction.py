import pandas as pd
import numpy as np
import sys
from enum import Enum

class Lab_Values(Enum):
	VL = 1
	L = 2
	N = 3
	H = 4
	VH = 5
class Location_Values(Enum):
	ED = 1
	NURSE = 2
	ICU =3
	STEPDN = 4
class Lab_Features(Enum):
    SystolicBP = 1
    DiastolicBP = 2
    HeartRate = 3
    RespiratoryRate = 4
    Temperature = 5
    PulseOx = 6
    FIO2 = 18
    OxygenFlow = 19
    BUN = 7
    Procalcitonin = 8
    WBC = 9
    Bands = 10
    Lactate = 11
    Platelet = 12
    Creatinine = 13
    MAP = 14
    BiliRubin = 15
    CReactiveProtein =16
    SedRate = 17
class Binary_Features(Enum):
    InfectionFlag=1
    InflammationFlag=2
    OrganFailure=3


class State:
	def __init__(self, feature, value):
		self.feature = feature
		self.value = value
	def describe(self):
		return "(" + self.feature + "," + str(self.value) + ")"
	def __eq__(self, other):
		if self.feature == other.feature and self.value == other.value:
			return True
		return False
	def __hash__(self):
		return hash((self.feature,self.value))

class StateInterval:
	def __init__(self, feature, value, start, end):
		self.feature = feature
		self.value = value
		self.start = start
		self.end = end
	def __gt__(self, state2):
		return self.start > state2.start
	def describe(self):
		return "(" + self.feature + "," + str(self.value) + "," + str(self.start) + "," + str(self.end) + ")"
	def find_relation(self, s2):
		if self.end < s2.start:
			return 'b'
		if self.start <= s2.start and s2.start <= self.end:
			return 'c'

# abstracts the values for a feature based on the whole data
def abstraction_alphabet(f1, f0):							
	lab_values = pd.concat([f1, f0])
	VL_range = np.percentile(lab_values[np.isfinite(lab_values)],10)
	L_range = np.percentile(lab_values[np.isfinite(lab_values)],25)
	N_range = np.percentile(lab_values[np.isfinite(lab_values)],75)
	H_range = np.percentile(lab_values[np.isfinite(lab_values)],90)
	VH_range = np.percentile(lab_values[np.isfinite(lab_values)],100)
	f1[f1<VL_range] = "VL"
	f1[(f1>=VL_range) & (f1<L_range)] = "L"
	f1[(f1>=L_range) & (f1<N_range)] = "N"
	f1[(f1>=N_range) & (f1<H_range)] = "H"
	f1[(f1>=H_range) & (f1<=VH_range)] = "VH"
	f0[f0<VL_range] = "VL"
	f0[(f0>=VL_range) & (f0<L_range)] = "L"
	f0[(f0>=L_range) & (f0<N_range)] = "N"
	f0[(f0>=N_range) & (f0<H_range)] = "H"
	f0[(f0>=H_range) & (f0<=VH_range)] = "VH"
	return f1, f0

# Gets abstracted values for a feature for a patient and returns the state intervals generated
def state_generation(abstracted_lab_values, feature):
	state_intervals = []
	previous_value = np.nan
	state_start = np.nan
	state_end = np.nan
	for i,val in abstracted_lab_values.iterrows():
		if pd.notnull(val[feature]) and pd.isnull(previous_value):
			previous_value = val[feature]
			state_start = val['MinutesFromArrival']
			state_end = val['MinutesFromArrival']
		elif pd.notnull(val[feature]) and (val[feature]==previous_value):
			state_end = val['MinutesFromArrival']
		elif pd.notnull(val[feature]) and (val[feature]!=previous_value):
			state_intervals.append(StateInterval(feature,previous_value,state_start,state_end))
			previous_value = val[feature]
			state_start = val['MinutesFromArrival']
			state_end = val['MinutesFromArrival']
	if pd.notnull(previous_value) and pd.notnull(state_end) and pd.notnull(state_start):
		state_intervals.append(StateInterval(feature,previous_value,state_start,state_end))
	return state_intervals

# Gets a sequence of data (for one patient) and returns the MSS
def MultivariateStateSequence(sequence_data):					
	MSS = []
	for f in Lab_Features:
		MSS.extend(state_generation(sequence_data, f.name))
	for f in Binary_Features:
		MSS.extend(state_generation(sequence_data, f.name))
	MSS.extend(state_generation(sequence_data, 'CurrentLocationTypeCode'))
	MSS.sort(key=lambda x: x.start)
	return MSS

# Find the index of state intervals in an MSS with same feature and value of state
def state_find_matches(mss, state, fi):							
	match = []
	for i in range (fi, len(mss)):
		if state.feature == mss[i].feature and state.value == mss[i].value:
			match.append(i)
	return match

# A recursive function that determines whether a sequence contains a pattern or not, based on DEFINITION 2
def MSS_contains_Pattern(mss, p, i, fi, prev_match):			
	if i >= len(p.states):
		return True, prev_match
	same_state_index = state_find_matches(mss, p.states[i], fi)
	for fi in same_state_index:
		flag = True
		for pv in range(0,len(prev_match)):
			if prev_match[pv].find_relation(mss[fi]) != p.relation[pv][i]:
				flag = False
				break
		if flag:
			prev_match.append(mss[fi])
			contains, seq = MSS_contains_Pattern(mss, p, i+1, 0, prev_match)
			if contains:
				return True, seq
			else:
				del prev_match[-1]
	return False, np.nan

# Determines whether a state interval is recent or not, based on DEFINITION 3
def recent_state_interval(mss, j, g):			
	if mss[len(mss)-1].end - mss[j].end <= g:
		return True
	flag = False
	for k in range(j+1, len(mss)):
		if mss[j].feature == mss[k].feature:
			flag = True
	if not flag:
		return True
	return False

def get_index_in_sequence(mss, e):				
	for i in range(0,len(mss)):
		if mss[i] == e:
			return i
	return -1

def sequences_containing_state(RTPlist, new_s):
	p_RTPlist = []
	for z in RTPlist:
		for e in z:
			if e.feature == new_s.feature and e.value == new_s.value:
				p_RTPlist.append(z)
				break
	return p_RTPlist

def find_all_frequent_states(D, support):
	freq_states = []
	for f in Lab_Features:
		for v in Lab_Values:
			state = State(f.name,v.name)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	for f in Binary_Features:
		for v in (0,1):
			state = State(f.name,v)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	for v in Location_Values:
		state = State('CurrentLocationTypeCode', v.name)
		if len(sequences_containing_state(D, state)) >= support:
			freq_states.append(state)
	return freq_states