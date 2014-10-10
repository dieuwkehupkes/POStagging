"""
A class to efficiently compute the best sequence of tags for a
sentence given an HMM model, using the Viterbi algorithm.
"""

import numpy


class Viterbi:
    """
    Functions to compute the Viterbi parse of a sentence
    """
    def __init__(self, hmm2):
        """
        :param hmm2: A second order hidden markov model
        :type hmm2: HMM2
        """
        self.hmm = hmm2

    def compute_best_parse(self, sequence):
        """
        Compute the viterbi parse for the sequence.
        """
        # rewrite this in matrixnotation
        # rewrite this using logs
        words = sequence.split()
        best_sequence = []
        N = len(self.hmm.tagIDs)
        length = len(words)
        dynamic_table = numpy.zeros(shape=(length, N, N), dtype=numpy.float64)
        backpointers = numpy.zeros(shape=dynamic_table.shape, dtype=object)

        # base case
        dynamic_table[0, -2, :] = self.hmm.transition[-1, -2, :] * self.hmm.emission[:, self.hmm.wordIDs[words[0]]]

        # work further through the sentence
        for position in xrange(1, length):
            wordID = self.hmm.wordIDs[words[position]]
            probs = numpy.transpose(dynamic_table[position-1, :, :] * self.hmm.transition[:, :, :].transpose(2, 0, 1), (0, 2, 1))
            probs_max = probs.max(axis=2)
            argmax1 = probs.argmax(axis=2)
            dynamic_table[position, :, :] = (self.hmm.emission[:, wordID, numpy.newaxis] * probs_max).transpose()

            # set backpointers (could I maybe also do this in the matrix)
            for tagID1 in xrange(N):
                argmax_prob = argmax1[tagID1]

                # create the backpointers, I would like to add this to an extra dimension of the current matrix so that
                # I can do it in one go
                for tagID2 in xrange(N):
                    argmax = argmax_prob[tagID2]
                    backpointer = (position-1, argmax, tagID2)
                    backpointers[position, tagID2, tagID1] = backpointer

        # last tag
        for tagID1 in xrange(N):
            for tagID2 in xrange(N):
                dynamic_table[-1, tagID1, tagID2] *= self.hmm.transition[tagID1, tagID2, -1]

        best_prob = dynamic_table[-1, :, :].max()
        i, j = numpy.unravel_index(dynamic_table[-1, :, :].argmax(), dynamic_table[-1, :, :].shape)
        # maybe put this in another function
        best_sequence.append(self.hmm.tagIDs_reversed[j])
        backpointer = (-1, i, j)
        for i in xrange(length-1):
            best_sequence.append(self.hmm.tagIDs_reversed[backpointer[1]])
            backpointer = backpointers[backpointer]

        best_sequence.reverse()

        return best_prob, best_sequence

    def traverse_backpointers():
        pass
