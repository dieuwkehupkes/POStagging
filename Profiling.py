from ForwardBackward import ForwardBackward as FB
import pickle

tags = set(['LET','LID', 'VZ', 'WW','TSW','ADJ','N','VG','BW','TW','SPEC(e)', 'VNW','VZ+LID','WW+VNW']) 

hmm = pickle.load(open('statenvertaling.hmm.pickle', 'r'))

s = 'inden beginne schiep godt den hemel , ende de aerde .\n'

training = FB(s, hmm)
training.compute_all_forward_probabilities()
training.compute_all_backward_probabilities()
