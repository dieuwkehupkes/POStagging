"""
The class implements an HMM model, several methods of smoothing are available
blablabla
"""

import sys
import copy
import numpy

class HMM2:
	"""
	Description of the class
	"""
	def __init__(self, transition_probabilities, emission_probabilities, tagIDs, wordIDs):
		self.emission = emission_probabilities
		self.transition = transition_probabilities
		self.tagIDs = tagIDs
		self.wordIDs = wordIDs
	
	def compute_probability(self, sequence, tags):
		"""
		Compute the probability of a tagged sequence.
		:param tagged_sequence: a list of (word, tag) tuples
		"""
		tags = ['###','$$$'] + tags + ['###']
		prob = 1.0
		#compute emission probabilities
		for i in xrange(len(sequence)):
			wordID = self.wordIDs[sequence[i]]
			tagID = self.tagIDs[tags[i+2]]
			try:
				prob = prob * self.emission[tagID,wordID]
			except IndexError:
				# Except will possibly be used when the tagger
				# is extended to work for new files
				print "No lexical probability available"
				prob = prob *self.get_smoothed_emission(tag,word)
				return 0
		#compute transition probabilities
		for i in xrange(2,len(tags)):
			tag1, tag2, tag3 = self.tagIDs[tags[i-2]], self.tagIDs[tags[i-1]], self.tagIDs[tags[i]]
			try:
				prob = prob * self.transition[tag1][tag2][tag3]
			except IndexError:
				# Except will possibly be used when the tagger
				# is extended to work for new files
				print "No transition probability available"
				prob = prob * self.get_smoothed_transition(tag1, tag2, tag3)
				return 0
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
				tagID = self.tagIDs[tag]
				probs[(pos,tagID)] = probs.get((pos,tagID),0.0) + prob
#			if sequence[1] == 'N':
				#print "new probability of word", s[1], "being N:", probs[(1,self.tagIDs['N'])]
		#compute totals for each position
		totals, e_count = {}, numpy.zeros(shape=self.emission.shape,dtype=numpy.float64)
		for position in xrange(len(s)):
			wordID = self.wordIDs[s[position]]
			totals[position] = sum([probs[x] for x in probs.keys() if x[0] == position])
			for tag in tags:
				tagID = self.tagIDs[tag]
				e_count[tagID, wordID] += (probs[(position, tagID)]/totals[position])
		return e_count
		
	def print_trigrams(self):
		import itertools
		tags_iterator = itertools.product(self.tagIDs.keys(),repeat=3)
		for trigram in tags_iterator:
			t1, t2, t3 = trigram
			print t1, '\t\t', t2, '\t\t', t3, '\t\t', self.transition[self.tagIDs[t1], self.tagIDs[t2],self.tagIDs[t3]]
		return
	
	def print_lexicon(self):
		for word in self.wordIDs.keys():
			pass
		raise NotImplementedError
		return
