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
		training.compute_all_forward_probabilities()
		training.compute_all_backward_probabilities()
		training.compute_all_products()
		training.compute_all_sums()
		training.compute_all_position_sums()
		expected_counts_fb = training.compute_expected_counts(tags)
		expected_counts_bf = hmm.expected_counts_brute_forse(s, tags)
		assert expected_counts_fb.keys() == expected_counts_bf.keys(), "keys forward backward: %s\n\n keys brute force: %s\n" % (expected_counts_fb.keys(), expected_counts_bf.keys())
		for key in expected_counts_fb:
			assert expected_counts_fb[key].keys() == expected_counts_bf[key].keys(), "keysposition %i expected counts forward backward: %s, expected counts brute forse: %s" % (key, expected_counts_fb[key].keys(), expected_counts_bf[key].keys())
		#	getcontext().prec = 10
			for tag in expected_counts_fb[key]:
				assert abs(expected_counts_fb[key][tag] - expected_counts_bf[key][tag]) < 1e-30, "expected_counts_fb: %f, expected_counts bf: %f" % (expected_counts_fb[key][tag], expected_counts_bf[key][tag])
				print expected_counts_fb[key][tag],'\n', expected_counts_bf[key][tag], '\n\n'
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
		trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test1')
		tags = set(['LID','VZ','N','WW'])
		trans_dict = generator.transition_dict_add_alpha(1, trans_dict, tags)
		word_dict = {'de': 1, 'man': 1, 'heeft': 1, 'een': 1, 'huis':1}
		lex_dict = generator.lexicon_dict_add_unlabeled(word_dict, lex_dict, tags)
		hmm = generator.make_hmm(trans_dict, lex_dict)
		os.remove('test1')
		return hmm
	
	def test_consistency_prob_model(self):
		"""
		Test if hmm probs sum to one
		"""
		hmm = self.make_toy_hmm()
		for key in hmm.emission:
			assert float(sum(hmm.emission[key].values())) == 1.0, "probabilities for tag %s do not sum to 1 but to %f, %s" % (key, sum(hmm.emission[key].values()),hmm.emission[key])
		for tag in hmm.transition:
			for tag2 in hmm.transition[tag]:
				s = sum(hmm.transition[tag][tag2].values())
				assert float(s) == 1.0, "probabilities for trigrams starting with %s %s do not sum up to 1 but to %f, %s" % (tag, tag2, s, hmm.transition[tag].values())
		return

	def test_all(self):
		"""
		Run all tests
		"""
		self.test_correctness_expected_counts1()
		self.test_consistency_prob_model()
		return
	
if __name__ == '__main__':
	T = Test()
	T.test_all()

