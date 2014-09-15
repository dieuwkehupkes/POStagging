from HMMgenerator import HMM2_generator as gen
from ForwardBackward import ForwardBackward as FB
from HMM2 import HMM2 as hmm
import copy
from decimal import *

#tags = set(['LID', 'VZ', 'WW','ADJ','N','VG','BW','VNW']) 
tags = set(['LET','LID', 'VZ', 'WW','TSW','ADJ','N','VG','BW','TW','SPEC(e)', 'VNW','VZ+LID','WW+VNW']) 
precision = 1000

generator = gen()
#trans_dict, lex_dict = generator.get_hmm_dicts_from_file('../../Data/StatenvertalingParallel/Test/test.1637.tags.gold')
trans_dict, lex_dict = generator.get_hmm_dicts_from_file('../../Data/StatenvertalingParallel/Test/test.1637.tags.gold')
print "created dictionaries from file"
trans_dict = generator.transition_dict_add_alpha(Decimal('1'), trans_dict, tags)
print "smoothed transition dictionary"
words = generator.get_words_from_file('../../Data/StatenvertalingParallel/temp')
lex_dict_smoothed = generator.lexicon_dict_add_unlabeled(words, lex_dict, tags)
print "smoothed lexical dictionary"
hmm = generator.make_hmm(trans_dict, lex_dict_smoothed)

lex = copy.deepcopy(lex_dict_smoothed)

for i in xrange(10):
	print "start iteration %i" % i
	#f = open('../../Data/StatenvertalingParallel/statenvertaling.1000.tok.1637.lower','r')
	f = open('../../Data/StatenvertalingParallel/temp','r')
	for line in f:
		training = FB(line, hmm, tags, precision)
		#training.compute_all_backward_probabilities()
		expected_counts = training.compute_expected_counts(tags)
		if line == 'inden beginne schiep godt den hemel , ende de aerde .\n':
			p = zip([x for x in xrange(11)],'VZ+LID N WW SPEC(e) LID N LET VG LID N LET'.split())
			for pair in p:
				position, tag = pair
				"print expected counts"
				print "P(%i, %s): %f" % (position, tag, expected_counts[position][tag])
			print '\n\n'
		lex = training.update_lexical_dict(lex, expected_counts)
	f.close()
	hmm = generator.make_hmm(trans_dict, lex)

	

