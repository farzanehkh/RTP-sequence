import pandas as pd
import numpy as np

import TemporalAbstraction
import RTPmining

class PatternInterval:
	def __init__(self, pattern, start, end):
		self.pattern = pattern
		self.start = start
		self.end = end
	def __gt__(self, state2):
		return self.start > state2.start
	def describe(self):
		return "(" + self.pattern.describe() + str(self.start) + "," + str(self.end) + ")"
	def find_relation(self, s2):
		if self.end < s2.start:
			return 'b'
		if self.start <= s2.start and s2.start <= self.end:
			return 'c'

def create_pattern_sequence(MSS, patterns):
	new_data = []
	for i in range(len(MSS)):
		pattern_seq = []
		for p in patterns:
			pattern_seq.extend(find_all_patterns(MSS[i], p))
		pattern_seq.sort(key=lambda x: x.start)
		binary_matrix = convert_sequence_to_matrix(pattern_seq, patterns)
		new_data.append(binary_matrix.tolist())
	return new_data

def convert_sequence_to_matrix(pattern_seq, patterns):
	times = [p.start for p in pattern_seq]
	times.extend([p.end for p in pattern_seq])
	times = list(set(times))
	binary_matrix = np.zeros((len(times),len(patterns)))
	times.sort()
	map_pi = {}
	for pi in pattern_seq:
		for i in range(len(patterns)):
			if patterns[i] == pi.pattern:
				pattern_index = i
				break
		map_pi.setdefault(pattern_index,[]).append(pi)
	for t in range(len(times)):
		for key, value in map_pi.items():
			for v in value:
				# 1. Binary vector representation is 1 ONLY during the occurance of pattern
				# if (v.start <= times[t] and v.end > times[t]) or (v.start == v.end and v.start == times[t]):
				# 	binary_matrix[t,key] = 1
				# 	break
				# 2. Binary vector representation is 1 from the end of the occurance of pattern
				# if (v.end <= times[t]) or (v.start == v.end and v.start == times[t]):
				# 	binary_matrix[t,key] = 1
				# 	break
				# 3. Binary vector representation is 1 ONLY after the occurance of pattern
				# if (v.end < times[t]):
				# 	binary_matrix[t,key] = 1
				# 	break
				# 4. Binary vector representation shows the number of occurrances of a pattern in the past
				if (v.end <= times[t]):
					binary_matrix[t,key] += 1
	return binary_matrix

def find_all_patterns(mss, p):
	pattern_seq = []
	contains, mapping, mapping_index = MSS_contains_Pattern(mss, p, 0, 0, [], [])
	while contains:
		pattern_seq.append(PatternInterval(p, mapping[0].start, mapping[-1].end))
		contains, mapping, mapping_index = MSS_contains_Pattern(mss, p, 0, mapping_index[-1]+1, [],[])
	# for p in pattern_seq:
	# 	print p.describe()
	return pattern_seq

# A recursive function that determines if an MSS contains a pattern and returns the mapping and index of mapping
def MSS_contains_Pattern(mss, p, i, next_i, prev_match, prev_match_index):			
	if i >= len(p.states):
		return True, prev_match, prev_match_index
	same_state_index = TemporalAbstraction.state_find_matches(mss, p.states[i], next_i)
	for fi in same_state_index:
		flag = True
		for pv in range(0,len(prev_match)):
			if prev_match[pv].find_relation(mss[fi]) != p.relation[pv][i]:
				flag = False
				break
		if flag:
			prev_match.append(mss[fi])
			prev_match_index.append(fi)
			contains, seq, seq_index = MSS_contains_Pattern(mss, p, i+1, next_i+1, prev_match, prev_match_index)
			if contains:
				return True, seq, seq_index
			else:
				del prev_match[-1]
				del prev_match_index[-1]
	return False, np.nan, np.nan