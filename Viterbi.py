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
        # Maybe I should rewrite this to use logs instead of probs
        words = sequence
        best_sequence = []
        N = len(self.hmm.tagIDs)
        length = len(words)
        dynamic_table = numpy.zeros(shape=(length, N, N), dtype=numpy.float64)

        backpointers = numpy.zeros(shape=dynamic_table.shape, dtype=numpy.int)

        # base case
        dynamic_table[0, -2, :] = self.hmm.transition[-1, -2, :] * self.hmm.emission[:, self.hmm.wordIDs[words[0]]]

        # work further through the sentence
        for position in xrange(1, length):
            wordID = self.hmm.wordIDs[words[position]]
            probs = dynamic_table[position-1, :, :] * self.hmm.transition[:, :, :].transpose(2, 0, 1)
            
            # set backpointers to highest probability indices
            # I compute both max and argmax, can't that be done more efficiently?
            backpointers[position, :, :] = probs.argmax(axis=1)

            # compute maximum values, update next row of dynamic table
            max_probs = probs.max(axis=1)
            dynamic_table[position, :, :] = (self.hmm.emission[:, wordID, numpy.newaxis] * max_probs).transpose()

        # last tag
        dynamic_table[-1, :, :] *= self.hmm.transition[:, :, -1]

        # probability best sequence
        tagID1, tagID2 = divmod(dynamic_table[-1, :, :].argmax(), N)
        backpointer = (-1, tagID1, tagID2)
        best_prob = dynamic_table[backpointer]

        # loop back trough backpointers to find best sequence
        # Maybe I should put this in another function
        best_sequence.append(self.hmm.tagIDs_reversed[tagID2])
        # backpointer = (-1, tagID1, tagID2)
        for position in reversed(xrange(length-1)):
            best_sequence.append(self.hmm.tagIDs_reversed[backpointer[1]])
            backpointer = (position, backpointers[position+1, tagID2, tagID1], tagID1)
            tagID1, tagID2 = backpointer[1], backpointer[2]

        best_sequence.reverse()

        return best_prob, best_sequence
