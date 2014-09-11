from HMMgenerator import HMM2_generator as gen
from ForwardBackward import ForwardBackward as FB
from HMM2 import HMM2 as hmm
import copy

tags = set(['LID','VZ','$$$','###','N','V'])
tags_ns = set(['LID','VZ','N','V'])
precision = 100

generator = gen()
trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test')
words = generator.get_words_from_file('test_unlabeled')
trans_dict = generator.transition_dict_add_alpha(0.1, trans_dict, tags)
lex_dict_smoothed = generator.emission_dict_add_alpha(2, lex_dict, words)
hmm = generator.make_hmm(trans_dict, lex_dict)


lex = copy.deepcopy(lex_dict_smoothed)

for i in xrange(3):
	f = open('test_unlabeled','r')

	for line in f:
		training = FB(line, hmm, tags, precision)
		expected_counts = training.compute_expected_counts(tags_ns)
		#print '\nposition sums:', training.position_sums, '\n\n\n\n'
		lex = training.update_lexical_dict(lex, expected_counts)
	f.close()
	hmm = generator.make_hmm(trans_dict, lex)

