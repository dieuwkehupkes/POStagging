"""

Implement parallel processing
"""

from IterativeTraining import Train
from IterativeTraining import CombinedDataset

# variables and files
tags = set(['LET', 'LID', 'VZ', 'WW', 'TSW', 'ADJ', 'N', 'VG', 'BW', 'TW', 'SPEC(e)', 'VNW', 'VZ+LID', 'WW+VNW', 'SPEC()'])
labeled = '../../Data/StatenvertalingParallel/Test/test.1637.tags.gold'
unlabeled = '../../Data/StatenvertalingParallel/temp'
syntactic_model = '../../Data/Lassy/lassy.trigrams'
evaluation = 'evaluation.temp'
lexical_smoothing_ratio = 0.1
transition_smoothing = 1.0
multiply_labeled = 100

D = CombinedDataset(labeled, unlabeled, syntactic_model, lexical_smoothing_ratio, tags)
T = Train(D)
hmm = D.hmm

T.train_and_test(hmm, 10, multiply_labeled, evaluation)
