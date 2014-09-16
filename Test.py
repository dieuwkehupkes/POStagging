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
		hmm = self.make_toy_hmm()
		tags = set(['LID','VZ','N','WW'])
		s = "de man heeft een huis"
		training = ForwardBackward(s,hmm,tags)

		#training.compute_all_forward_probabilities()
		#for key in training.forward:
		#	print key, training.forward[key]
		expected_counts_fb = training.compute_expected_counts()
		expected_counts_bf = hmm.expected_counts_brute_forse(s, tags)
		assert expected_counts_fb.keys() == expected_counts_bf.keys(), "keys forward backward: %s\n\n keys brute force: %s\n" % (expected_counts_fb.keys(), expected_counts_bf.keys())
		for key in expected_counts_fb:
			assert expected_counts_fb[key].keys() == expected_counts_bf[key].keys(), "keysposition %i expected counts forward backward: %s, expected counts brute forse: %s" % (key, expected_counts_fb[key].keys(), expected_counts_bf[key].keys())
		#	getcontext().prec = 10
			for tag in expected_counts_fb[key]:
				assert abs(expected_counts_fb[key][tag] - expected_counts_bf[key][tag]) < 1e-30, "expected_counts_fb: %f, expected_counts bf: %f" % (expected_counts_fb[key][tag], expected_counts_bf[key][tag])
				#assert expected_counts_fb[key][tag] == expected_counts_bf[key][tag], "expected_counts_fb: %f, expected_counts bf: %f" % (expected_counts_fb[key][tag], expected_counts_bf[key][tag])
		return
	
	def make_toy_hmm(self):
		"""
		Create a toy HMM
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
		self.test_correctness_expected_counts1()
		return
	
if __name__ == '__main__':
	T = Test()
	T.test_all()

