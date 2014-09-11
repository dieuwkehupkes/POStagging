from HMMgenerator import HMM2_generator as gen
from ForwardBackward import ForwardBackward as FB
from HMM2 import HMM2 as hmm
import copy

tags = set(['LID','VZ','$$$','###','N','V'])
tags_ns = set(['LID','VZ','N','V'])
precision = 100

generator = gen()
trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test')
trans_dict = generator.transition_dict_add_alpha(0.5, trans_dict, tags)
words = generator.get_words_from_file('test_unlabeled') 
#lex_dict_smoothed = generator.emission_dict_add_alpha(0.5, lex_dict, words)
lex_dict_smoothed = generator.lexicon_dict_add_unlabeled('test_unlabeled', lex_dict, tags_ns)
hmm = generator.make_hmm(trans_dict, lex_dict)


lex = copy.deepcopy(lex_dict_smoothed)

for i in xrange(20):
	f = open('test_unlabeled','r')

	for line in f:
		training = FB(line, hmm, tags, precision)
		expected_counts = training.compute_expected_counts(tags_ns)
		#print '\nposition sums:', training.position_sums, '\n\n\n\n'
		lex = training.update_lexical_dict(lex, expected_counts)
		if line == 'de jongen rent naar huis\n':
			for p in [(0, 'LID'),(1,'N'), (2,'V'),(3,'VZ'),(4,'N')]:
				position, tag = p
				print "P(%i, %s): %f" % (position, tag, expected_counts[position][tag])
			print '\n\n'
	f.close()
	hmm = generator.make_hmm(trans_dict, lex)

