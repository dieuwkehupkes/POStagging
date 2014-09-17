"""
A class to efficiently compute expected counts using the forward-backward algorithm

DON'T FORGET TO INCLUDE ENDING AND STARTING COUNTS IN EXPECTED COUNTS!
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
	def __init__(self, sentence, hmm2, precision=100):
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
		#self.forward = numpy.zeros(shape=(len_sentence,self.N, self.N))
		#self.backward = numpy.zeros(shape=(len_sentence,self.N, self.N))
		self.forward = {}
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
		#I think I should be able to do this in one go with some matrix multiplication
		#I will figure that out later once I have made everything into matrices
		for i in xrange(len(self.sentence)):
			wordID = self.wordIDs[self.sentence[i]]
			for tagID in xrange(self.N):
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
		try:
			probability = self.sums[position,tagID]/self.position_sums[position]
		except KeyError:
			raise ValueError("Compute sums and position sums")
		except ValueError:
			raise ValueError("Forward and backward probabilities should be computed before computing tag probabilities")
		except ZeroDivisionError:
			probability = 0
		except:
			#print '\n'.join(['tag: %s, prob: %f' % (tp[1], self.sums[tp]) for tp in self.sums if tp[0] == position])
			print 'position', position, '\ntag', tagID, '\nsums[position,tags]', self.sums[position, tagID], '\n\nposition_sums[position', self.position_sums[position]
			raise ZeroDivisionError
		return probability
	
	def get_smoothed_prob(self,tag,sentence):
		"""
		Get smoothed probability for a word-tag pair that did not
		occur in the dictionary. This function is mainly included
		for more flexibility later, as smoothing probabilities is likely
		to be conducted before starting EM. Currently the returned
		'smoothed' probability is just 0.
		"""
		return 0

	
	def forward_probability(self, position, tagID1, tagID2):
		"""
		Compute forward probabilities. Note that although this is in
		principle a recursive algorithm, it only executes well when
		all required values are computed iteratively in the right order.
		If a value required to compute a forward probability is not computed
		prior to the to be computed forward probability, it will be taken
		as 0, and the forward probability will therefore evaluate to a lower
		number that it actually is.
		#should I do something about that?
		Note that the position starts counting at 0.
		"""
		#if forward probability is already computed, return
		if (position, tagID1, tagID2) in self.forward:
			return self.forward[(position, tagID1, tagID2)]
		# tag $$$ or tag ### never occur for words
		if tagID1 == self.ID_start or tagID1 == self.ID_end:
			self.forward[(position,tagID1,tagID2)] = Decimal('0.0')
			return Decimal('0.0')

		wordID = self.wordIDs[self.sentence[position]]
		
		#base case of the recursion
		if position == 0:
			if tagID2 != self.ID_start:
				self.forward[(position,tagID1, tagID2)] = 0
				return 0
			else:
				try:
					prob = self.hmm.emission[tagID1, wordID]*self.hmm.transition[-1,-2,tagID1]
				except KeyError:
					#shouldn't be needed now
					print "lexical probability has no entry"
					prob = self.get_smoothed_prob(tagID1,wordID) * self.hmm.transition[-1,-2,tagID1]
				self.forward[(position,tagID1,tagID2)] = prob
				return prob

		if tagID2 == self.ID_start or tagID2 == self.ID_end:
			#if position is not 0, tag of previous word cannot be $$$ or ###
			self.forward[(position,tagID1,tagID2)] = Decimal(0.0)
			return Decimal(0.0)

		#position further in the sentence
		try:
			e_prob = self.hmm.emission[tagID1,wordID]
		except KeyError:
			#shouldn't be needed now
			print "no lexical entry for", tagID1, wordID
			e_prob = self.get_smoothed_prob(tagID1, wordID)
		#marginalise over possible tags
		sum_alpha = 0
		for tagID in xrange(self.N +1):
			try:
				new_alpha = self.forward[(position-1,tagID2,tagID)]
			except KeyError:
				print "define iteration differently"
				new_alpha = self.forward(position-1,tagID2,tagID)
			new_transition = self.hmm.transition[tagID,tagID2,tagID1]
			sum_alpha += new_alpha*new_transition
		e_prob = e_prob * sum_alpha		
		self.forward[(position,tagID1,tagID2)] = e_prob
		return e_prob

	def compute_all_forward_probabilities(self):
		"""
		Iterative algorithm to compute all forward
		probabilities.
		"""
		#In principle are there some things that can be excluded or forehand
		# such as "$$$" "$$$" X, maybe I should hard code skipping these cases
		
		#Loop over all combinations of tags and positions
		for position, tagID1, tagID2 in itertools.product(xrange(len(self.sentence)),xrange(self.N+1), xrange(self.N+1)):
			#compute forward probability
			self.forward_probability(position,tagID1,tagID2)
		#transform to matrix
		forward_matrix = numpy.zeros(shape=(len(self.sentence),self.N+2,self.N+2), dtype = Decimal)
		for position, tagID1, tagID2 in self.forward:
			forward_matrix[position, tagID1, tagID2] = self.forward[(position, tagID1, tagID2)]
		self.forward = forward_matrix
		return

	def backward_probability(self, position, tagID1, tagID2):
		"""
		Recursively compute backward probabilities. Make use of 
		previously computed probabilies to avoid recomputing.
		Counting starts at 0.
		"""
		#print 'compute bacward probability for', position, tagID1, tagID2
		#check if backward probability was previously computed
		if (position, tagID1, tagID2) in self.backward:
			return self.backward[(position,tagID1,tagID2)]
		#base case of the recursion
		if position == len(self.sentence)-1:
			try:
				prob = self.hmm.transition[tagID2,tagID1,-1] #I am not entirely sure whether this is correct
			except KeyError:
				prob = 0
			self.backward[(position, tagID1, tagID2)] = prob
			return prob
		next_wordID = self.wordIDs[self.sentence[position+1]]
		sum_betas = 0
		for tagID in xrange(self.N):
			try:
				lex_prob = self.hmm.emission[tagID,next_wordID]
			except KeyError:
				#shouldn't be needed now
				print "lexical probability missing"
				lex_prob = self.get_smoothed_prob(tagID, next_word)
			#print "lexprob %s %s: %f" % (tag, next_word, lex_prob)
			try:
				beta_prob = self.backward[(position+1,tagID,tagID1)]
			except KeyError:
				print "Backward probs, change iteration order"
				beta_prob = self.backward_probability(position+1,tagID,tagID1)

			#print "beta probability (%i, %s, %s): %f" % (position, tag, tag1, beta_prob)
			try:
				trigram_prob = self.hmm.transition[tagID2][tagID1][tagID]
			except KeyError:
				#shouldn't be needed now
				print "trigram probability missing"
				trigram_prob = 0
			#print "trigram probability %s %s %s: %f" % (tagID2, tagID1, tag, trigram_prob)
			prob_all = lex_prob*beta_prob*trigram_prob
			#print "prob_all: %f" % prob_all
			sum_betas += prob_all
		#print "sum_betas:", sum_betas
		self.backward[(position,tagID1,tagID2)] = sum_betas
		#print "backward_prob(%i,%s,%s) = %f" % (position, tagID1, tagID2, sum_betas)
		return sum_betas
	
	def compute_all_backward_probabilities(self):
			"""
			Compute all backward probabilities for the sentence
			"""
			for position, tagID1, tagID2 in itertools.product(reversed(xrange(len(self.sentence))),xrange(self.N),xrange(self.N+1)):
					#compute backward probability
					self.backward_probability(position,tagID1,tagID2)
			backward_matrix = numpy.zeros(shape=(len(self.sentence),self.N+2,self.N+2), dtype = Decimal)
			for position, tagID1, tagID2 in self.backward:
				backward_matrix[position, tagID1, tagID2] = self.backward[(position, tagID1, tagID2)]
			self.backward = backward_matrix
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
					
