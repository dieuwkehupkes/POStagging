"""
I should write a test module to see if the
forward and backward probabilities are really computed correctly
"""

import os
from ForwardBackward import *
from HMM2 import *
from HMMgenerator import *

class Test:
	"""
	Test suite for semi supervised EM module.
	"""
	
	def  test_correctness_expected_counts1(self):
		"""
		Test if forward backward algorithm computes
		the same expected counts as bruteforce algorithm
		"""
		hmm = self.toy_hmm_smoothed()
		tags = set(['LID','VZ','N','WW'])
		s = "de man heeft een huis"
		training = ForwardBackward(s,hmm,tags)

		#training.compute_all_forward_probabilities()
		#for key in training.forward:
		#	print key, training.forward[key]
		expected_counts_fb = training.compute_expected_counts()
		expected_counts_bf = hmm.expected_counts_brute_forse(s, tags)
		print expected_counts_bf[training.tagIDs['N'],training.wordIDs['man']]
		print expected_counts_fb[training.tagIDs['N'],training.wordIDs['man']]
		print expected_counts_bf - expected_counts_fb
		print abs(expected_counts_bf - expected_counts_fb) < 1e-30
		return
	
	def test_generation(self):
		"""
		For a manually worked out example, test if the
		transition probabilities found by the hmm-generator
		are correct.
		"""
		hmm = self.toy_hmm()
		transition_matrix_man = numpy.zeros(shape=hmm.transition.shape, dtype=Decimal)
		transition_matrix_man += Decimal('0.0')
		tagIDs = hmm.tagIDs
		for t1,t2,t3 in [('$$$','LID','N'),('###','$$$','LID'),('WW','VZ','N'),('WW','LID','N'), ('N','VZ','LID'),('VZ','LID','N'),('VZ','N','###')]:
			transition_matrix_man[tagIDs[t1],tagIDs[t2],tagIDs[t3]] = Decimal('1.0')
		transition_matrix_man[tagIDs['LID'],tagIDs['N'],tagIDs['WW']] = Decimal('0.5')
		transition_matrix_man[tagIDs['LID'],tagIDs['N'],tagIDs['###']] = Decimal('2.0')/Decimal('6.0')
		transition_matrix_man[tagIDs['LID'],tagIDs['N'],tagIDs['VZ']] = Decimal('1.0')/Decimal('6.0')
		transition_matrix_man[tagIDs['N'],tagIDs['WW'],tagIDs['LID']] = Decimal('2.0')/Decimal('3.0')
		transition_matrix_man[tagIDs['N'],tagIDs['WW'],tagIDs['VZ']] = Decimal('1.0')/Decimal('3.0')
		assert numpy.array_equal(transition_matrix_man, hmm.transition), transition_matrix_man == hmm.transition
		return
	
	def test_HMM2_compute_probability(self):
		"""
		Test the "compute probability" function of
		the HMM2 class.
		"""
		hmm = self.toy_hmm()
		s = "de man heeft een huis".split()
		tags = "LID N WW LID N".split()
		man_prob = Decimal('16.0')/Decimal('15876.0')
		prob = hmm.compute_probability(s,tags)
		assert man_prob == prob
		return
	
	def toy_hmm(self):
		"""
		Create a toy HMM with unsmoothed transition and
		lexical probabilities.
		"""
		f = open('test1','w')
		f.write("de\tLID\nman\tN\nloopt\tWW\nnaar\tVZ\nhuis\tN\n\nde\tLID\nman\tN\nheeft\tWW\neen\tLID\nhond\tN\nmet\tVZ\neen\tLID\nstaart\tN\n\nhet\tLID\nhuis\tN\nheeft\tWW\neen\tLID\ndeur\tN")
		f.close()
		generator = HMM2_generator()
		words_labeled = generator.labeled_make_word_list('test1')
		words_unlabeled = {'de': 1, 'man': 1, 'loopt': 1, 'naar': 1, 'huis':1}
		all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
		tags = set(['LID','VZ','N','WW'])
		trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test1',tags, all_words)
		hmm = generator.make_hmm(trans_dict, lex_dict)
		return hmm
		
	
	def toy_hmm_smoothed(self):
		"""
		Create a toy HMM with smoothed lexical probabilities.
		"""
		f = open('test1','w')
		f.write("de\tLID\nman\tN\nloopt\tWW\nnaar\tVZ\nhuis\tN\n\nde\tLID\nman\tN\nheeft\tWW\neen\tLID\nhond\tN\nmet\tVZ\neen\tLID\nstaart\tN\n\nhet\tLID\nhuis\tN\nheeft\tWW\neen\tLID\ndeur\tN")
		f.close()
		generator = HMM2_generator()
		words_labeled = generator.labeled_make_word_list('test1')
		words_unlabeled = {'de': 1, 'man': 1, 'heeft': 1, 'een': 1, 'huis':1}
		all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
		tags = set(['LID','VZ','N','WW'])
		trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test1',tags, all_words)
		trans_dict = generator.transition_dict_add_alpha(0.5, trans_dict)
		lex_dict = generator.lexicon_dict_add_unlabeled(words_unlabeled, lex_dict)
		hmm = generator.make_hmm(trans_dict, lex_dict)
		os.remove('test1')
		return hmm
	
	def test_all(self):
		"""
		Run all tests
		"""
		self.test_generation()
		self.test_HMM2_compute_probability()
		#self.test_correctness_expected_counts1()
		return
	
if __name__ == '__main__':
	T = Test()
	T.test_all()

