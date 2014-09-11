"""
A class to efficiently compute expected counts using the forward-backward algorithm

DON'T FORGET TO INCLUDE ENDING AND STARTING COUNTS IN EXPECTED COUNTS!
"""

from HMM2 import *
from HMMgenerator import *
from decimal import *

class ForwardBackward:
	"""
	Compute the expected counts for tags in
	the sentence. Initialise with an HMM model,
	a set of possible tags and a sentence.
	"""
	def __init__(self, sentence, hmm2, possible_tags, precision=100):
		"""
		:param sentence: A tokenised sentence, either a string or a list of words
		:param hmm2: A second order hmm model
		:type hmm2: HMM2
		:param possible_tags: the tags possible in the model
		:type possible_tags: set
		"""
		if isinstance(sentence,str):
			self.sentence = sentence.split()
		elif isinstance(sentence,list):
			self.sentence = sentence
		else:
			raise TypeError("Sentence should be of type string or list")
		self.hmm = hmm2
		self.tags = possible_tags
		self.transition_add_zero()
		self.forward = {}
		self.backward = {}
		getcontext().prec = precision
	
	def update_lexical_dict(self, lex_dict, expected_counts):
		"""
		Update the inputted lexical dictionary with the
		expected counts
		"""
		d = copy.deepcopy(lex_dict)
		for pos in expected_counts:
			for tag in expected_counts[pos]:
				try:
					d[tag][self.sentence[pos]] += Decimal(expected_counts[pos][tag])
				except KeyError:
					print tag, self.sentence[pos]
					print d[tag][self.sentence[pos]]
					print expected_counts[pos][tag]
				
		return d

	def transition_add_zero(self):
		"""
		Add entries for trigrams with 0 probability to the
		transition dictionary.
		"""
		for tag1 in self.tags:
			self.hmm.transition[tag1] = self.hmm.transition.get(tag1, {})
			for tag2 in self.tags:
				self.hmm.transition[tag1][tag2] = self.hmm.transition[tag1].get(tag2,{})
				for tag3 in self.tags:
					self.hmm.transition[tag1][tag2][tag3] = self.hmm.transition[tag1][tag2].get(tag3,Decimal(0))
		return

	def compute_expected_counts(self,tags):
		"""
		Compute the counts for every tag at every possible position
		"""
		expected_counts = {}
		self.compute_all_forward_probabilities()
		self.compute_all_backward_probabilities()
		self.compute_all_products()
		self.compute_all_sums()
		self.compute_all_position_sums()
		for i in xrange(len(self.sentence)):
			expected_counts[i] = {}	
			for tag in tags:
				prob = self.compute_tag_probability(i,tag)
				expected_counts[i][tag] = prob
		return expected_counts
	
	def compute_tag_probability(self, position, tag):
		"""
		Compute the probability that the HMM model asigns tag
		to word at position.
		It is assumed that all forward and backward probabilities
		for the sentence are already computed
		"""
		i = tag
		try:
			probability = self.sums[(position,tag)]/self.position_sums[position]
		except KeyError:
			raise ValueError("Compute sums and position sums")
		except ValueError:
			raise ValueError("Forward and backward probabilities should be computed before computing tag probabilities")
		except ZeroDivisionError:
			probability = 0
		except:
			print '\n'.join(['tag: %s, prob: %f' % (tp[1], self.sums[tp]) for tp in self.sums if tp[0] == position])
			#print 'position', position, '\ntag', tag, '\nsums[position,tags]', self.sums[(position, tag)], '\n\nposition_sums[position', self.position_sums[position]
			raise ZeroDivisionError
			probability = 0
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

	
	def forward_probability(self, position, tag1, tag2):
		"""
		Recursively compute forward probabilities. Make use of the
		forward probabilities dictionary to avoid recomputing already
		computed things.
		Note that the position starts counting at 0.
		"""
		#if forward probability is already computed, return
		if (position, tag1, tag2) in self.forward:
			return self.forward[(position, tag1, tag2)]
		#base case of the recursion
		word = self.sentence[position]
		if position == 0:
			if tag2 == "$$$":
				try:
					prob = self.hmm.emission[tag1][word]*self.hmm.transition["###"]["$$$"][tag1]
				except KeyError:
					prob = self.get_smoothed_prob(tag1,word) * self.hmm.transition['###']['$$$'][tag1]
				self.forward[(position,tag1,tag2)] = prob
				return prob
			#tag2 is always start of sentence, otherwise prob = 0
			self.forward[(position,tag1,tag2)] = 0
			return 0
		#position further in the sentence
		try:
			prob = self.hmm.emission[tag1][word]
		except KeyError:
			prob = self.get_smoothed_prob(tag1, word)
		#marginalise over possible tags
		sum_alpha = 0
		for tag in self.tags:
			new_alpha = self.forward_probability(position-1,tag2,tag)
			sum_alpha += (self.forward_probability(position-1, tag2,tag)*self.hmm.transition[tag][tag2][tag1])
		prob = prob * sum_alpha		
		self.forward[(position,tag1,tag2)] = prob
		return prob

	def compute_all_forward_probabilities(self):
		"""
		Iterative algorithm to compute all forward
		probabilities.
		"""
		#In principle are there some things that can be excluded or forehand
		# such as "$$$" "$$$" X, maybe I should hard code skipping these cases
		len(self.sentence)
		for position in xrange(len(self.sentence)):
			for tag1 in self.tags:
				for tag2 in self.tags:
					#compute forward probability
					self.forward_probability(position,tag1,tag2)
		return

	def backward_probability(self, position, tag1, tag2):
		"""
		Recursively compute backward probabilities. Make use of 
		previously computed probabilies to avoid recomputing.
		Counting starts at 0.
		"""
		#check if backward probability was previously computed
		if (position, tag1, tag2) in self.backward:
			return self.backward[(position,tag1,tag2)]
		#base case of the recursion
		if position == len(self.sentence)-1:
			prob = self.hmm.transition[tag2][tag1]["###"] #I am not entirely sure whether this is correct
			self.backward[(position, tag1, tag2)] = prob
			return prob
		next_word = self.sentence[position+1]
		sum_betas = 0
		for tag in self.tags:
			try:
				lex_prob = self.hmm.emission[tag][next_word]
			except KeyError:
				lex_prob = self.get_smoothed_prob(tag, next_word)
			beta_prob = self.backward_probability(position+1,tag,tag1)
			trigram_prob = self.hmm.transition[tag2][tag1][tag]
			prob_all = lex_prob*beta_prob*trigram_prob
			sum_betas += prob_all
		self.backward[(position,tag1,tag2)] = sum_betas
		return sum_betas
	
	def compute_all_backward_probabilities(self):
			"""
			Compute all backward probabilities for the sentence
			"""
			for position in reversed(xrange(len(self.sentence))):
				for tag1 in self.tags:
					for tag2 in self.tags:
						#compute backward probability
						self.backward_probability(position,tag1,tag2)
			return
	
	def compute_all_sums(self):
		"""
		After computing the forward and backward probabilities,
		compute all sums required to compute the expected counts.
		This function can only be used AFTER computing the forward-
		and backward probabilities.
		"""
		try:
			self.products == {}
		except AttributeError:
			self.compute_all_products()
		self.sums = {}
		for pos in xrange(len(self.sentence)):
			for i in self.tags:
				self.sums[(pos,i)] = 0
				for j in self.tags:
					self.sums[(pos,i)] += self.products[(pos,i,j)]
		return self.sums
	
	def compute_all_position_sums(self):
		"""
		Compute the total probability mass going to a tag position.
		Used for normalisation.
		"""
		try:
			self.sums == {}
		except AttributeError:
			self.compute_all_sums
		self.position_sums = {}
		for pos in range(len(self.sentence)):
			self.position_sums[pos] = 0
			for tag in self.tags:
				self.position_sums[pos] += self.sums[(pos,tag)]
		return self.position_sums
					
	def compute_all_products(self):
		"""
		Compute the products of all forward and backward probabilities
		with the same variables.
		"""
		self.products = {}
		for pos in xrange(len(self.sentence)):
			for i in self.tags:
				for j in self.tags:
					self.products[(pos,i,j)] = self.forward[(pos,i,j)]*self.backward[(pos,i,j)]
		return self.products
					

def find_counts_brute_forse(hmm, sentence, tags):
	import itertools
	probs = {}
	i = 0
	s = sentence.split()
	sequence_iterator = itertools.product(tags,repeat=len(s))
	for sequence in sequence_iterator:
		i += 1
		prob = hmm.compute_probability(s, list(sequence))
		for pos in xrange(len(sequence)):
			tag = sequence[pos]
			probs[(pos,tag)] = probs.get((pos,tag),0) + prob
	return probs


if __name__ == '__main__':
	f = 'test'
	generator = HMM2_generator()
	trans_dict, lex_dict = generator.get_hmm_dicts_from_file(f)
	tags = set(['LID','VZ','$$$','###','N','V'])
	trans_dict = generator.transition_dict_add_alpha(0.1, trans_dict, tags)
	lex_dict = generator.emission_dict_add_alpha(0.1, lex_dict, (['de','man','heeft','een','huis']))
	hmm = generator.make_hmm(trans_dict, lex_dict)
	#hmm.print_trigrams()
	#hmm.print_lexicon()
	s = 'de man heeft een huis'
	#tags = set(['LID','VZ','$$$','###','N','V', 'ADJ', 'ADV','TW', 'VG','LET','TSW', 'VNW'])
	training = ForwardBackward(s, hmm, tags)
	training.compute_all_backward_probabilities()
	training.compute_all_forward_probabilities()
	training.compute_all_sums()
	training.compute_all_position_sums()
#	print "non-zero forward probabilities:"
#	print '\n'.join(["alpha%s: %f" % (item[0], item[1]) for item in training.forward.items() if item[1]!=0])
#	print "\nnon-zero backward probabilities:"
#	print '\n'.join(["beta%s: %f" % (item[0], item[1]) for item in training.backward.items() if item[1]!=0])
#	print "\nnon-zero products"
#	print '\n'.join(["products%s: %f" % (item[0], item[1]) for item in training.products.items() if item[1]!=0])
#	print "\nnon-zero sums"
#	print '\n'.join(["sums%s: %f" % (item[0], item[1]) for item in training.sums.items() if item[1]!=0])
	expected_counts = training.compute_expected_counts(tags)
#	for position in expected_counts:
#		print '\n', position, 
#		for tag in expected_counts[position]:
#			print '\t', tag, expected_counts[position][tag]
	probs = find_counts_brute_forse(hmm,s,tags)
	total1 = sum([probs[tag] for tag in probs.keys() if tag[0]==1])
	print 'P(1,V)', probs[(1,'V')]/total1

