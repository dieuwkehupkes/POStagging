"""
A class for iterative semi-supervised training.
explain better
MOVE COMPUTE EXPECTED COUNTS FUNCTION TO HMM2 CLASS
"""

# from HMMgenerator import HMM2_generator as gen
from HMM2 import HMM2
from ForwardBackward import ForwardBackward
from HMMgenerator import HMM2_generator as gen
import numpy
import pickle


class CombinedDataset:
    """
    A class representing a dataset consisting of a labeled
    and an unlabeled part.
    """
    def __init__(self, labeled, unlabeled, syntactic_model, lexical_smoothing_ratio, tags):
        self.labeled = labeled
        self.unlabeled = unlabeled
        generator = gen()
        words_labeled = generator.labeled_make_word_list(labeled)
        words_unlabeled = generator.unlabeled_make_word_list(unlabeled)
        all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
        wordIDs = generator.generate_lexicon_IDs(all_words)
        trans_dict, tagIDs = pickle.load(open(syntactic_model, 'r'))
        generator.wordIDs = wordIDs
        generator.tagIDs = tagIDs
        lex_dict = generator.get_lexicon_from_file(labeled, tagIDs, wordIDs)
        lex_dict_smoothed = generator.weighted_lexicon_smoothing(lex_dict, words_unlabeled, ratio=lexical_smoothing_ratio)
        self.lex_dict = lex_dict
        self.hmm = generator.make_hmm(trans_dict, lex_dict_smoothed, tagIDs, wordIDs)

    def train(self, start, iterations, scaling):
            """
            semi-supervised training
            more explanation
            """
            training = Train(self.lex_dict, self.unlabeled)
            training.train(self.hmm, 1, scaling)


class Train:
    """
    Given a labeled and an unlabeled set of data and a set of parameters
    train an hmm2 model semi supervised. Functions are also available for
    evaluating the quality of the tags after every iteration, if evaluation
    labels are available.
    """
    def __init__(self, combined_dataset):
        self.D = combined_dataset
        self.lex_basis = self.D.lex_dict
        self.unlabeled = self.D.unlabeled

    def train(self, start, iterations, scaling):
        """
        Train the model using the labeled and unlabeled corpus.
        Some more explanation.
        """
        hmm = start
        lexicon_basis = scaling * self.lex_basis
        for i in xrange(iterations):
            hmm = self.iteration(hmm, lexicon_basis)
        return hmm

    def iteration(self, hmm, labeled_counts):
        """
        Do one iteration of training.
        """
        f = open(self.unlabeled, 'r')
        sum_expected_counts = numpy.zeros(shape=hmm.emission.shape, dtype=numpy.float64)
        for line in f:
            training = ForwardBackward(line, hmm)
            expected_counts = training.compute_expected_counts()
            sum_expected_counts = training.update_lexical_dict(expected_counts, sum_expected_counts)
        new_lexicon = training.update_lexical_dict(sum_expected_counts, labeled_counts)
        f.close()
        new_hmm = HMM2(hmm.transition, new_lexicon, hmm.tagIDs, hmm.wordIDs)
        return new_hmm
