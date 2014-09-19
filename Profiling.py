from ForwardBackward import ForwardBackward as FB
import pickle

tags = set(['LET', 'LID', 'VZ', 'WW', 'TSW', 'ADJ', 'N', 'VG', 'BW', 'TW', 'SPEC(e)', 'VNW', 'VZ+LID', 'WW+VNW'])

hmm = pickle.load(open('statenvertaling.hmm.pickle', 'r'))

s = 'inden beginne schiep godt den hemel , ende de aerde .\n'

training = FB(s, hmm)
expected_counts = training.compute_expected_counts()

pickle.dump(expected_counts, open('expected_counts_float.pickle', 'w'))
