"""
A class to efficiently compute expected counts using the forward-backward algorithm

TODO:
compute backward probabilities with matrix multiplications
compute expected counts with matrix multiplications
to save memory: try whether using floats maybe suffices
replace all matrix indexing with 'take'
"""

from HMM2 import *
from HMMgenerator import *
from decimal import Decimal
import itertools

class ForwardBackward:
	"""
	Compute the expected counts for tags in
	the sentence. Initialise with an HMM model,
	a set of possible tags and a sentence.
	"""
	def __init__(self, sentence, hmm2, precision=200):
		"""
		:param sentence: A tokenised sentence, either a string or a list of words
		:param hmm2: A second order hmm model
		:type hmm2: HMM2
		:param possible_tags: the tags possible in the model
		:type possible_tags: set
		"""
		self.sentence = sentence.split()
		self.hmm = hmm2
		self.tagIDs = hmm2.tagIDs
		self.wordIDs = hmm2.wordIDs
		self.N = len(self.tagIDs) -2
		self.ID_start = self.tagIDs['$$$']
		self.ID_end = self.tagIDs['###']
		self.backward = {}
		getcontext.prec = precision
	
	def update_lexical_dict(self, lex_dict, expected_counts):
		"""
		Update the inputted lexical dictionary with the
		expected counts
		"""
		lex_dict += expected_counts
		return lex_dict

	def compute_expected_counts(self, expected_counts = None):
		"""
		Compute the counts for every tag at every possible position
		"""
		#initialise an empty expected counts matrix or use inputted one
		if expected_counts:
			pass
		else:
			expected_counts = numpy.zeros(shape=self.hmm.emission.shape, dtype = Decimal)
			expected_counts += Decimal('0.0')
		self.compute_all_forward_probabilities()
		self.compute_all_backward_probabilities()
		self.compute_all_products()
		self.compute_all_sums()
		self.compute_all_position_sums()
		for i in xrange(len(self.sentence)):
			wordID = self.wordIDs[self.sentence[i]]
			for tagID in xrange(self.N+2):
				prob = self.compute_tag_probability(i,tagID)
				expected_counts[tagID, wordID] += prob
		return expected_counts
	
	def compute_tag_probability(self, position, tagID):
		"""
		Compute the probability that the HMM model asigns tag
		to word at position.
		It is assumed that all forward and backward probabilities
		for the sentence are already computed
		"""
		probability = self.sums[position,tagID]/self.position_sums[position]
		return probability
	
	def compute_all_forward_probabilities(self):
		"""
		Iterative algorithm to compute all forward
		probabilities.
		"""
		forward = numpy.zeros(shape=(len(self.sentence),self.N+2,self.N+2), dtype = Decimal)

		#compute the base case of the recursion (position=0)
		wordID = self.wordIDs[self.sentence[0]]
		forward[0] = numpy.transpose(self.hmm.emission[:,wordID]*self.hmm.transition[-1])

		for pos in xrange(1, len(self.sentence)):
			wordID = self.wordIDs[self.sentence[pos]]
			M = (numpy.take(forward, [pos-1],axis=0)*self.hmm.transition.transpose(2,1,0)).sum(axis=2)
			forward[pos] = M*self.hmm.emission[:,wordID,numpy.newaxis]

		self.forward = forward
		return

	def compute_all_backward_probabilities(self):
		"""
		Compute all backward probabilities for the sentence
		"""
		backward = numpy.zeros(shape=(len(self.sentence), self.N+2, self.N+2), dtype = Decimal)

		#Compute the values for the base case of the recursion
		backward[len(self.sentence)-1]= self.hmm.transition[:,:,-1].transpose()

		#fill the rest of the matrix
		for pos in reversed(xrange(len(self.sentence)-1)):
			next_wordID = self.wordIDs[self.sentence[pos+1]]
			for tagID1 in xrange(self.N+2):
				for tagID2 in xrange(self.N+2):
					backward[pos,tagID1, tagID2] = (self.hmm.emission[:,next_wordID]*backward[pos+1, :, tagID1]*self.hmm.transition[tagID2, tagID1, :]).sum()
		self.backward = backward
		return
	
	def compute_all_sums(self):
		"""
		After computing the forward and backward probabilities,
		compute all sums required to compute the expected counts.
		This function can only be used AFTER computing the forward-
		and backward probabilities.
		"""
		self.sums = self.products.sum(axis=2)
		return self.sums
	
	def compute_all_position_sums(self):
		"""
		Compute the total probability mass going to a tag position.
		Used for normalisation.
		"""
		self.position_sums = self.sums.sum(axis=1)
		return self.position_sums
					
	def compute_all_products(self):
		"""
		Compute the products of all forward and backward probabilities
		with the same variables.
		"""
		self.products = numpy.multiply(self.forward, self.backward)
		return self.products
					
