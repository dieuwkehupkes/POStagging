"""
Test Module
I should write something that tests if the smoothed probabilities are alright
"""


import os
from HMMgenerator import HMM2_generator
import numpy


class Test:
    """
    Test suite for semi supervised EM module.
    """

    def test_correctness_expected_counts1(self):
        """
        Test if forward backward algorithm computes
        the same expected counts as bruteforce algorithm
        """
        hmm = self.toy_hmm_smoothed()
        tags = set(['LID', 'VZ', 'N', 'WW'])
        s = "de man heeft een huis"
        expected_counts_fb = hmm.compute_expected_counts(s)
        expected_counts_bf = hmm.expected_counts_brute_forse(s, tags)
        assert numpy.all(abs(expected_counts_bf - expected_counts_fb) < 1e-15), expected_counts_bf == expected_counts_fb
        return

    def test_generation(self):
        """
        For a manually worked out example, test if the
        transition probabilities found by the hmm-generator
        are correct.
        """
        hmm = self.toy_hmm()
        transition_matrix_man = numpy.zeros(shape=hmm.transition.shape, dtype=numpy.float64)
        tagIDs = hmm.tagIDs
        for t1, t2, t3 in [('$$$', 'LID', 'N'), ('###', '$$$', 'LID'), ('WW', 'VZ', 'N'), ('WW', 'LID', 'N'), ('N', 'VZ', 'LID'), ('VZ', 'LID', 'N'), ('VZ', 'N', '###')]:
            transition_matrix_man[tagIDs[t1], tagIDs[t2], tagIDs[t3]] = 1.0
        transition_matrix_man[tagIDs['LID'], tagIDs['N'], tagIDs['WW']] = 0.5
        transition_matrix_man[tagIDs['LID'], tagIDs['N'], tagIDs['###']] = 2.0 / 6.0
        transition_matrix_man[tagIDs['LID'], tagIDs['N'], tagIDs['VZ']] = 1.0 / 6.0
        transition_matrix_man[tagIDs['N'], tagIDs['WW'], tagIDs['LID']] = 2.0 / 3.0
        transition_matrix_man[tagIDs['N'], tagIDs['WW'], tagIDs['VZ']] = 1.0 / 3.0
        assert numpy.array_equal(transition_matrix_man, hmm.transition), transition_matrix_man == hmm.transition
        return

    def test_HMM2_compute_probability(self):
        """
        Test the "compute probability" function of
        the HMM2 class.
        """
        hmm = self.toy_hmm()
        s = "de man heeft een huis".split()
        tags = "LID N WW LID N".split()
        man_prob = 16.0 / 15876.0
        prob = hmm.compute_probability(s, tags)
        assert abs(man_prob - prob) < 1e-10
        return

    def test_viterbi(self):
        """
        Test de function finding the Viterbi parse.
        """
        hmm = self.toy_hmm()
        s = "de man heeft een huis"
        prob, sequence = hmm.compute_best_sequence(s)
        assert abs(prob - 16/15876.0) < 1e-10
        assert sequence == ['LID', 'N', 'WW', 'LID', 'N']
        return

    def test_viterbi2(self):
        """
        Test de viterbi parse function using the
        compute_probability function.
        """
        hmm = self.toy_hmm_smoothed()
        s = "de man heeft een huis"
        prob, sequence = hmm.compute_best_sequence(s)
        prob2 = hmm.compute_probability('de man heeft een huis'.split(), sequence)
        assert prob == prob2
        return

    def toy_hmm(self):
        """
        Create a toy HMM with unsmoothed transition and
        lexical probabilities.
        """
        f = open('test1', 'w')
        f.write("de\tLID\nman\tN\nloopt\tWW\nnaar\tVZ\nhuis\tN\n\nde\tLID\nman\tN\nheeft\tWW\neen\tLID\nhond\tN\nmet\tVZ\neen\tLID\nstaart\tN\n\nhet\tLID\nhuis\tN\nheeft\tWW\neen\tLID\ndeur\tN")
        f.close()
        generator = HMM2_generator()
        words_labeled = generator.labeled_make_word_list('test1')
        words_unlabeled = {'de': 1, 'man': 1, 'loopt': 1, 'naar': 1, 'huis': 1}
        all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
        tags = set(['LID', 'VZ', 'N', 'WW'])
        trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test1', tags, all_words)
        hmm = generator.make_hmm(trans_dict, lex_dict, generator.tagIDs, generator.wordIDs)
        return hmm

    def toy_hmm_smoothed(self):
        """
        Create a toy HMM with smoothed lexical probabilities.
        """
        f = open('test1', 'w')
        f.write("de\tLID\nman\tN\nloopt\tWW\nnaar\tVZ\nhuis\tN\n\nde\tLID\nman\tN\nheeft\tWW\neen\tLID\nhond\tN\nmet\tVZ\neen\tLID\nstaart\tN\n\nhet\tLID\nhuis\tN\nheeft\tWW\neen\tLID\ndeur\tN")
        f.close()
        generator = HMM2_generator()
        words_labeled = generator.labeled_make_word_list('test1')
        words_unlabeled = {'de': 2, 'man': 1, 'heeft': 1, 'een': 1, 'huis': 1}
        all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
        tags = set(['LID', 'VZ', 'N', 'WW'])
        trans_dict, lex_dict = generator.get_hmm_dicts_from_file('test1', tags, all_words)
        trans_dict = generator.transition_dict_add_alpha(0.5, trans_dict)
        lex_dict = generator.weighted_lexicon_smoothing(lex_dict, words_unlabeled, ratio=0.5)
        # lex_dict = generator.lexicon_dict_add_unlabeled(words_unlabeled, lex_dict)
        hmm = generator.make_hmm(trans_dict, lex_dict, generator.tagIDs, generator.wordIDs)
        os.remove('test1')
        return hmm

    def test_all(self):
        """
        Run all tests
        """
        self.test_generation()
        self.test_HMM2_compute_probability()
        self.test_correctness_expected_counts1()
        self.test_viterbi()
        self.test_viterbi2()
        return

if __name__ == '__main__':
    T = Test()
    T.test_all()
