"""
The class implements an HMM model, several methods of smoothing are available
blablabla
"""

import sys
import copy
from decimal import *

class HMM2:
	"""
	Description of the class
	"""
	def __init__(self, transition_probabilities, emission_probabilities):
		self.emission = emission_probabilities
		self.transition = transition_probabilities
#		getcontext.prec = 5000
	
	def compute_probability(self, sequence, tags):
		"""
		Compute the probability of a tagged sequence.
		:param tagged_sequence: a list of (word, tag) tuples
		"""
		tags = ['###','$$$'] + tags + ['###']
		prob = Decimal('1.0')
		#compute emission probabilities
		for i in xrange(len(sequence)):
			word = sequence[i]
			tag = tags[i+2]
			try:
				prob = prob * self.emission[tag][word]
			except KeyError:
				return 0
				prob = prob *self.get_smoothed_emission(tag,word)
		#compute transition probabilities
		for i in xrange(2,len(tags)):
			tag1, tag2, tag3 = tags[i-2], tags[i-1], tags[i]
			try:
				prob = prob * self.transition[tag1][tag2][tag3]
			except KeyError:
				return 0
				prob = prob * self.get_smoothed_transition(tag1, tag2, tag3)
		return prob
	
	def get_smoothed_emission(self, tag, word):
		"""
		Smoothed probability if a word-tag pair does not
		occur in the lexicon. For future use, currently just
		returns 0.
		"""
		return 0
	
	def get_smoothed_transition(self, tag1, tag2, tag3):
		"""
		Smoothed transition probability for unseen trigrams.
		For future use, currently just returns 0.
		"""
		return 0
	
	def expected_counts_brute_forse(self, sentence, tags):
		import itertools
		probs = {}
		s = sentence.split()
		#generate all possible tag sequences
		sequence_iterator = itertools.product(tags, repeat=len(s))
		#find probability of each tag-position pair
		for sequence in sequence_iterator:
			prob = self.compute_probability(s, list(sequence))
			for pos in xrange(len(sequence)):
				tag = sequence[pos]
				probs[(pos,tag)] = probs.get((pos,tag),Decimal('0')) + prob
		#compute totals for each position
		totals, e_count = {}, {}
		for position in xrange(len(s)):
			e_count[position] = {}
			totals[position] = sum([probs[x] for x in probs.keys() if x[0] == position])
			for tag in tags:
				try:
					e_count[position][tag] = probs[(position,tag)]/totals[position]
				except ZeroDivisionError:
					e_count[position][tag] = 0
		return e_count	
		
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
	
