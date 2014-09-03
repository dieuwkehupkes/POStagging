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
		self.transition = transition_probabilities
	
	def compute_probability(self, sequence, tags):
		"""
		Compute the probability of a tagged sequence.
		:param tagged_sequence: a list of (word, tag) tuples
		"""
		tags = ['###','$$$'] + tags + ['###']
		prob = 1
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
	
