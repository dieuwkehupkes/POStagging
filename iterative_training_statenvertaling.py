from HMMgenerator import HMM2_generator as gen
from ForwardBackward import ForwardBackward as FB
import copy
import numpy

tags = set(['LET', 'LID', 'VZ', 'WW', 'TSW', 'ADJ', 'N', 'VG', 'BW', 'TW', 'SPEC(e)', 'VNW', 'VZ+LID', 'WW+VNW'])
labeled = '../../Data/StatenvertalingParallel/Test/test.1637.tags.gold'
unlabeled = '../../Data/StatenvertalingParallel/temp'

generator = gen()
words_labeled = generator.labeled_make_word_list(labeled)
words_unlabeled = generator.unlabeled_make_word_list(unlabeled)
all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
trans_dict, lex_dict = generator.get_hmm_dicts_from_file(labeled, tags, all_words)
print "created dictionaries from file"
trans_dict = generator.transition_dict_add_alpha(1.0, trans_dict)
print "smoothed transition dictionary"
lex_dict_smoothed = generator.lexicon_dict_add_unlabeled(words_unlabeled, lex_dict)
print "smoothed lexical dictionary"
hmm = generator.make_hmm(trans_dict, lex_dict_smoothed)

lex = copy.deepcopy(lex_dict_smoothed)

for i in xrange(20):
    print "start iteration %i" % i
    f = open(unlabeled, 'r')
    sum_expected_counts = numpy.zeros(shape=hmm.emission.shape, dtype=numpy.float64)
    for line in f:
        training = FB(line, hmm)
        expected_counts = training.compute_expected_counts()
        if line == 'inden beginne schiep godt den hemel , ende de aerde .\n':
            words = line.split()
            wordIDs = [hmm.wordIDs[x] for x in words]
            tags = 'VZ+LID N WW SPEC(e) LID N LET VG LID N LET'.split()
            tagIDs = [hmm.tagIDs[x] for x in tags]
            for i in xrange(11):
                wordID = wordIDs[i]
                tagID = tagIDs[i]
                print "P(%s, %s): %f" % (words[i], tags[i], expected_counts[tagID, wordID])
            print '\n\n'
            raw_input()
        sum_expected_counts = training.update_lexical_dict(expected_counts, sum_expected_counts)
    lex = training.update_lexical_dict(sum_expected_counts, lex_dict)
    f.close()
    hmm = generator.make_hmm(trans_dict, lex)
