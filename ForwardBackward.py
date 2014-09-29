"""
A class to efficiently compute expected counts using the forward-backward algorithm
"""

import numpy


class ForwardBackward:
    """
    Compute the expected counts for tags in
    the sentence. Initialise with an HMM model,
    a set of possible tags and a sentence.
    """
    def __init__(self, sentence, hmm2):
        """
        :param sentence: A tokenised sentence, either a string or a list of words
        :param hmm2: A second order hmm model
        :type hmm2: HMM2
        :param possible_tags: the tags possible in the model
        :type possible_tags: set
        """
        self.sentence = sentence.split()
        self.hmm = hmm2
        self.tagIDs = hmm2.tagIDs
        self.wordIDs = hmm2.wordIDs
        self.N = len(self.tagIDs)
        self.ID_start = self.tagIDs['$$$']
        self.ID_end = self.tagIDs['###']
        self.backward = {}

    def update_lexical_dict(self, lex_dict, expected_counts):
        """
        Update the inputted lexical dictionary with the
        expected counts
        """
        lex_dict += expected_counts
        return lex_dict

    def update_count_dict(self, expected_counts):
        """
        Update the counts in an inputted
        count matrix with the expected counts
        for this sentence.
        """
        raise NotImplementedError

    def compute_expected_counts(self):
        """
        Compute the counts for every tag at every possible position
        """
        # I don't know if this can maybe be done better with the word_indices
        expected_counts = numpy.zeros(shape=self.hmm.emission.shape, dtype=numpy.float64)

        # compute all required sums and probabilities
        self.compute_all_forward_probabilities()
        self.compute_all_backward_probabilities()
        self.compute_all_products()
        self.compute_all_sums()
        self.compute_all_position_sums()

        # compute expected counts
        word_tag_sums = self.sums / self.position_sums[:, numpy.newaxis]
        word_indices = [self.wordIDs[word] for word in self.sentence]
        expected_counts[:, word_indices] = word_tag_sums.transpose()

        return expected_counts

    # @profile
    def compute_all_forward_probabilities(self):
        """
        Iterative algorithm to compute all forward
        probabilities.
        """
        forward = numpy.zeros(shape=(len(self.sentence), self.N, self.N), dtype = numpy.float64)

        # compute the base case of the recursion (position=0)
        wordID = self.wordIDs[self.sentence[0]]
        forward[0] = numpy.transpose(self.hmm.emission.take(wordID, axis=1) * self.hmm.transition.take(-1, axis=0))

        # iteratively fill the forward matrix
        for pos in xrange(1, len(self.sentence)):
            wordID = self.wordIDs[self.sentence[pos]]
            M = (forward.take(pos - 1, axis=0) * self.hmm.transition.transpose(2, 1, 0)).sum(axis=2)
            forward[pos] = M * self.hmm.emission[:, wordID, numpy.newaxis]
        self.forward = forward
        return

    # @profile
    def compute_all_backward_probabilities(self):
        """
        Compute all backward probabilities for the sentence
        """
        backward = numpy.zeros(shape=(len(self.sentence), self.N, self.N), dtype = numpy.float64)

        # Compute the values for the base case of the recursion
        backward[len(self.sentence) - 1] = self.hmm.transition.take(-1, axis=2).transpose()

        # Fill the rest of the matrix
        for pos in reversed(xrange(len(self.sentence) - 1)):
            next_wordID = self.wordIDs[self.sentence[pos + 1]]
            M = (backward[pos + 1, :, :] * self.hmm.emission[:, next_wordID, numpy.newaxis]).transpose()
            fsums = self.hmm.transition.transpose(1, 0, 2) * M[:, numpy.newaxis, :]
            backward[pos] = fsums.sum(axis=2)

        self.backward = backward
        return

    def compute_all_sums(self):
        """
        After computing the forward and backward probabilities,
        compute all sums required to compute the expected counts.
        This function can only be used AFTER computing the forward-
        and backward probabilities.
        """
        self.sums = self.products.sum(axis=2)
        return self.sums

    def compute_all_position_sums(self):
        """
        Compute the total probability mass going to a tag position.
        Used for normalisation.
        """
        self.position_sums = self.sums.sum(axis=1)
        return self.position_sums

    def compute_all_products(self):
        """
        Compute the products of all forward and backward probabilities
        with the same variables.
        """
        self.products = numpy.multiply(self.forward, self.backward)
        return self.products
