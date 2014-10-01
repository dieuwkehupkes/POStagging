"""
The class implements an HMM model, I should implement some smoothing models.
"""

import numpy
from Viterbi import Viterbi


class HMM2:
    """
    Description of the class
    """
    def __init__(self, transition_probabilities, emission_probabilities, tagIDs, wordIDs):
        self.emission = emission_probabilities
        self.transition = transition_probabilities
        self.tagIDs = tagIDs
        self.tagIDs_reversed = self.get_reversed_tagIDs()
        self.wordIDs = wordIDs

    def get_reversed_tagIDs(self):
        """
        Create a matrix in which for each number the
        corresponding tag can be found.
        """
        reversed_tagIDs = numpy.empty(len(self.tagIDs), dtype='|S8')
        for tag in self.tagIDs:
            ID = self.tagIDs[tag]
            reversed_tagIDs[ID] = tag
        return reversed_tagIDs

    def compute_probability(self, sequence, tags):
        """
        Compute the probability of a tagged sequence.
        :param tagged_sequence: a list of (word, tag) tuples
        """
        tags = ['###', '$$$'] + tags + ['###']
        prob = 1.0

        # compute emission probabilities
        for i in xrange(len(sequence)):
            wordID = self.wordIDs[sequence[i]]
            tagID = self.tagIDs[tags[i+2]]
            prob = prob * self.emission[tagID, wordID]

        # compute transition probabilities
        for i in xrange(2, len(tags)):
            tag1, tag2, tag3 = self.tagIDs[tags[i - 2]], self.tagIDs[tags[i-1]], self.tagIDs[tags[i]]
            prob = prob * self.transition[tag1, tag2, tag3]

        return prob

    def compute_best_sequence(self, sequence):
        """
        Compute the optimal tag-sequence for the
        given sentence.
        """
        viterbi = Viterbi(self)
        best_parse = viterbi.compute_best_parse(sequence)
        return best_parse

    def compute_expected_counts(self, sequence):
        """
        I think I should maybe use the function here
        instead of in the ForwardBackward algorithm
        """
        raise NotImplementedError

    def get_smoothed_emission(self, tag, word):
        """
        Smoothed probability if a word-tag pair does not
        occur in the lexicon. For future use, currently just
        returns 0.
        """
        return 0

    def get_smoothed_transition(self, tag1, tag2, tag3):
        """
        Smoothed transition probability for unseen trigrams.
        For future use, currently just returns 0.
        """
        return 0

    def expected_counts_brute_forse(self, sentence, tags):
        import itertools
        probs = {}
        s = sentence.split()
        # generate all possible tag sequences
        sequence_iterator = itertools.product(tags, repeat=len(s))
        # find probability of each tag-position pair
        for sequence in sequence_iterator:
            prob = self.compute_probability(s, list(sequence))
            for pos in xrange(len(sequence)):
                tag = sequence[pos]
                tagID = self.tagIDs[tag]
                probs[(pos, tagID)] = probs.get((pos, tagID), 0.0) + prob

        # compute totals for each position
        totals, e_count = {}, numpy.zeros(shape=self.emission.shape, dtype=numpy.float64)
        for position in xrange(len(s)):
            wordID = self.wordIDs[s[position]]
            totals[position] = sum([probs[x] for x in probs.keys() if x[0] == position])
            for tag in tags:
                tagID = self.tagIDs[tag]
                e_count[tagID, wordID] += (probs[(position, tagID)]/totals[position])
        return e_count

    def print_trigrams(self):
        import itertools
        tags_iterator = itertools.product(self.tagIDs.keys(), repeat=3)
        for trigram in tags_iterator:
            t1, t2, t3 = trigram
            print t1, '\t\t', t2, '\t\t', t3, '\t\t', self.transition[self.tagIDs[t1], self.tagIDs[t2], self.tagIDs[t3]]
        return

    def print_lexicon(self):
        for word in self.wordIDs.keys():
            pass
        raise NotImplementedError
        return
