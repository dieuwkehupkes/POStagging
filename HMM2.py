"""
The class implements an HMM model, several methods of smoothing are available
blablabla
"""

import sys
import copy

class HMM2:
	"""
	Description of the class
	"""
	def __init__(self, transition_probabilities, emission_probabilities):
		self.emission = emission_probabilities
		self.transition_add_zero()
		self.transition = transition_probabilities
	
	
	def compute_probability(self, tagged_sequence):
		"""
		Compute the probability of a tagged sequence.
		:param tagged_sequence: a list of (word, tag) tuples
		"""
		s = [('###','###'), ('$$$','$$$')] + tagged_sequence + [('###','###')]
		prob = 1
		#compute emission probabilities
		try:
			for pair in s[2:-1]:
				prob = prob * self.emission[pair[1]][pair[0]]
		except KeyError:
			print "Not all words occurred in lexicon"
			return 0
		#compute transition probabilities
		try:
			for i in xrange(2,len(s)):
				tag1, tag2, tag3 = s[i-2][1], s[i-1][1], s[i][1]
				prob = prob * self.transition[tag1][tag2][tag3]
		except KeyError:
			print "Not all trigrams occur in model"
			print tag1, tag2,tag3
			return 0
		return prob
	
	
	def print_trigrams(self):
		for tag1 in self.transition:
			print tag1
			for tag2 in self.transition[tag1]:
				print '\t\t', tag2
				for tag3 in self.transition[tag1][tag2]:
					print '\t\t\t\t', tag3, '\t', self.transition[tag1][tag2][tag3]
		return
	
	def print_lexicon(self):
		for tag in self.emission:
			print tag
			for word in self.emission[tag]:
				print '\t\t\t', word, '\t', self.emission[tag][word]
		return
	
