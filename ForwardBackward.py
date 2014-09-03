"""
A class to efficiently compute expected counts using the forward-backward algorithm
"""

class ForwardBackward:
	"""
	Compute the expected counts for tags in
	the sentence. Initialise with an HMM model,
	a set of possible tags and a sentence
	"""
	def __init__(self, sentence, hmm2, possible_tags):
		self.sentence = sentence
		self.hmm = hmm2
		self.tags = possible_tags
		self.forward = {}
		self.backward = {}

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
					self.hmm.transition[tag1][tag2][tag3] = self.hmm.transition[tag1][tag2].get(tag3,0)
		return
	
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
		if position == 0:
			if tag2 = "$$$":
				prob = self.hmm.emission{tag1}{sentence[position]}*self.hmm.transition{"###"}{"$$$"}{tag1}
				self.forward[(position,tag1,tag2)] = prob
				return prob
			#tag2 is always start of sentence, otherwise prob = 0
			self.forward[(position,tag1,tag2)] = 0
			return 0
		prob = self.hmm.emission{tag1}{sentence[position]}
		sum_alphas = 0
		for tag in self.tags
			sum_alpha += (self.forward_probability(position-1, tag2,tag)*self.hmm.transition{tag2}{tag}{tag1})
		prob = prob * sum_alpha		
		self.forward[(position,tag1,tag2)] = prob
		return prob

	def compute_all_forward_probabilities(self):
		"""
		Iterative algorithm to compute all forward
		probabilities.
		"""
		for position in xrange(len(self.sentence))
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
		if (position, tag1, tag2) in self.bacward:
			return self.backward[(position,tag1,tag2)]
		#base case of the recursion
		if position == len(self.sentence):
			prob = self.hmm.transition{tag1}{tag2}{"###")} #I am not entirely sure whether this is correct
			self.backward[(position, tag1, tag2)] = prob
			return prob
		sum_betas = 0
		for tag in self.tags:
			lex_prob = self.hmm.emission{tag}{self.sentence[position+1]}
			beta_prob = self.backward_probability(position+1,tag,tag1)
			trigram_prob = self.transition{tag1}{tag2}{tag}
			prob_all = lex_prob*beta_prob*trigram_prob
			sum_betas += prob_all
		self.backward[(position,tag1,tag2)] = sum_betas
		return sum_betas
	
	def compute_all_backward_probabilities(self):
			for position in xrange(len(self.sentence),0,-1):
				for tag1 in self.tags:
					for tag2 in self.tags:
						#compute backward probability
						self.backward_probability(position,tag1,tag2)
			return
		raise NotImplementedError



