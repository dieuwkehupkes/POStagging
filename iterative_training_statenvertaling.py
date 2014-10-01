"""

Implement parallel processing
"""

from IterativeTraining import Train
from IterativeTraining import CombinedDataset
from ForwardBackward import ForwardBackward as FB

# variables and files
tags = set(['LET', 'LID', 'VZ', 'WW', 'TSW', 'ADJ', 'N', 'VG', 'BW', 'TW', 'SPEC(e)', 'VNW', 'VZ+LID', 'WW+VNW', 'SPEC()'])
syntactic = '../../Data/Lassy/lassy.train.spec1'
labeled = '../../Data/StatenvertalingParallel/Test/test.1637.tags.gold'
unlabeled = '../../Data/StatenvertalingParallel/temp'
syntactic_model = '../../Data/Lassy/lassy.trigrams'
lexical_smoothing_ratio = 0.1
transition_smoothing = 1.0
multiply_labeled = 100

D = CombinedDataset(labeled, unlabeled, syntactic_model, lexical_smoothing_ratio, tags)
T = Train(D)
hmm = D.hmm

for i in xrange(20):
    print "start iteration %i" % i
    hmm = T.train(hmm, 1, multiply_labeled)
    # hmm = T.iteration(hmm, lex_dict)

    line = 'inden beginne schiep godt den hemel , ende de aerde .\n'
    words = line.split()
    tags = 'VZ+LID N WW SPEC(e) LID N LET VG LID N LET'.split()
    wordIDs = [hmm.wordIDs[x] for x in words]
    training = FB(line, hmm)
    expected_counts = training.compute_expected_counts()
    tagIDs = [hmm.tagIDs[x] for x in tags]
    for i in xrange(11):
        wordID = wordIDs[i]
        tagID = tagIDs[i]
        print "P(%s, %s): %f" % (words[i], tags[i], expected_counts[tagID, wordID])
    # for tag in generator.tagIDs:
    #    print "P(%s, %s): %f" % ('godt', tag, expected_counts[generator.tagIDs[tag], wordIDs[3]])
    print '\n\n'
    i = raw_input('press Enter to continue, Q to quit ')
    if i == 'q' or i == 'Q':
        exit()
