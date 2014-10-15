"""
Functions to generate HMMs.
Add scaled smoothing lexicon dict (based on how often a word occured)
"""

from HMM2 import HMM2
import string
import numpy
import copy
from collections import Counter
import csv      # do something with this
import sys
import itertools

class HMM2_generator:
    """
    HMM2_generator is a class with functions to generate second
    order hidden markov models from text files.
    """

    def init_transition_matrix(self, tagIDs):
        """
        Transition probabilities are stored in a 3-dimensional
        matrix. The probabilities can be accessed with tagIDs.
        E.g., P(t3| t1, t2) can be found in T[t1, t2, t3].
        Initialise an empty transition matrix.

        :param tagIDs: a map from tagnames to IDs
        :type tagIDs: dictionary
        """
        N = len(tagIDs)
        transition_matrix = numpy.zeros(shape=(N, N, N), dtype=numpy.float64)
        return transition_matrix

    def init_lexicon_matrix(self, wordIDs, tagIDs):
        """
        Emission probabilities are stored in a 2-dimensional
        matrix. The probabilities can be accessed with tag-
        and wordIDs. E.g. P(w|t) can be found in E[t, w].
        Initialise an empty lexicon matrix.
        """
        nr_of_words = len(wordIDs)
        nr_of_tags = len(tagIDs)
        lexicon_matrix = numpy.zeros(shape=(nr_of_tags, nr_of_words), dtype=numpy.float64)
        return lexicon_matrix

    def generate_tag_IDs(self, tags):
        """
        Given a set of tags, generate a dictionary that
        assigns an ID to every tag. The tagID can be used
        to look up information in the transition and
        emission matrix.
        """
        IDs = [x for x in xrange(len(tags))]
        maxID = IDs[-1]
        self.tagIDs = dict(zip(tags, IDs))
        self.tagIDs['$$$'] = maxID + 1
        self.tagIDs['###'] = maxID + 2
        return self.tagIDs

    def generate_lexicon_IDs(self, words):
        """
        Generate a dictionary that stores the relation between
        words and emission matrix. The ID generated for a
       word is the index that can be used to look up the
        word in the emission matrix
        """
        self.wordIDs = dict(zip(words, [i for i in xrange(len(words))]))
        return self.wordIDs

    def find_tags(self, input_data):
        """
        Find all the tags occuring in the inputdata. Input data
        can be both a file name and a list with sentences.
        """
        data = self.get_data(input_data, delimiter=True)
        tags = set([item[1] for item in data if item != []])
        return tags

    def get_hmm_dicts_from_file(self, input_data, tags, words):
        """
        Generate a matrix containing trigram counts from a file.
        The beginning and end of the sentence are modelled with
        the beginning and end of sentence tags ('$$$' and "###',
        respectively).

        :param input_data:  A list with sentences or a file with labeled
                            sentences. If input is a file, every line of it
                            should contain a word and a tag separated by a tab.
                            New sentences should be delimited by new lines.
        :type input_data:   string or list
        :param tags:        A list or set of possible tags
        :param words:       A list or set of all words
        """
        # make list representation data.
        data = self.get_data(input_data, delimiter=True)

       # generate word and tagIDs 
        wordIDs = self.generate_lexicon_IDs(words)
        tagIDs = self.generate_tag_IDs(tags)
        ID_end, ID_start = tagIDs['###'], tagIDs['$$$']

        # initialise transition and emission matrix
        trigrams = self.init_transition_matrix(tagIDs)
        emission = self.init_lexicon_matrix(wordIDs, tagIDs)

        # initialisatie
        prev_tagID, cur_tagID = ID_end, ID_start

        # loop over lines in list:
        # Maybe this can be done faster or smarter
        for wordtag in data:
            try:
                # line contains a tag and a word
                word, tag = wordtag
                wordID, tagID = wordIDs[word], tagIDs[tag]
                trigrams[prev_tagID, cur_tagID, tagID] += 1
                emission[tagID, wordID] += 1
                prev_tagID = cur_tagID
                cur_tagID = tagID
            except ValueError:
                # end of sentence
                trigrams[prev_tagID, cur_tagID, ID_end] += 1
                trigrams[cur_tagID, ID_end, ID_start] += 1
                prev_tagID, cur_tagID = ID_end, ID_start

        # add last trigram if file did not end with white line
        if prev_tagID != ID_end:
            trigrams[prev_tagID, cur_tagID, ID_end] += 1
            trigrams[cur_tagID, ID_end, ID_start] += 1

        return trigrams, emission

    def get_lexicon_counts(self, input_data, tagIDs, wordIDs):
        """
        containing a word and a tag separated by a tab.

        :param input_data:  A list with sentences or a file with labeled
                            sentences. If input is a file, every line of it
                            should contain a word and a tag separated by a tab.
                            New sentences should be delimited by new lines.
                            should contain a word and a tag separated by a tab.
                            New sentences should be delimited by new lines.
        :type input_data:   string or list
        :param tagIDs:      A dictionary with IDs for all possible tags.
        :param wordIDs:     A dictionary with IDs for all possible words.
        """
        # Load data
        data = self.get_data(input_data, delimiter=True)

        # initialise emission matrix
        emission = self.init_lexicon_matrix(wordIDs, tagIDs)

        # generate counts for all words in the data
        counts = Counter([tuple(item) for item in data])

        # remove newlines
        del counts[()]

        # add counts to lexicon
        for wordtag in counts:
            word, tag = wordtag
            emission[tagIDs[tag], wordIDs[word]] = counts[wordtag]

        return emission

    def get_trigrams_from_file(self, input_data, tagIDs):
        """
        Generate a matrix with trigram counts from the input file.
        Use the tagIDs as indices for the different tags.
        containing lines with a word and a tag separated by a tab.

        :param input_data:  A list with sentences or a file with labeled
                            sentences. If input is a file, every line of it
                            should contain a word and a tag separated by a tab.
                            New sentences should be delimited by new lines.
        :type input_data:   string or list
        :param tagIDs:      A dictionary with IDs for all possible tags.
        :param wordIDs:     A dictionary with IDs for all possible words.
        """
        # get data
        data = self.get_data(input_data, delimiter='\t')

        # Initialisation
        trigrams = self.init_transition_matrix(tagIDs)
        ID_end, ID_start = tagIDs['###'], tagIDs['$$$']
        prev_tagID, cur_tagID = ID_end, ID_start        # beginning of sentence

        # loop over data
        # for line in data:
        for line in data:
            try:
                word, tag = line
                tagID = tagIDs[tag]
                trigrams[prev_tagID, cur_tagID, tagID] += 1.0
                prev_tagID = cur_tagID
                cur_tagID = tagID
            except ValueError:
                # end of sentence
                trigrams[prev_tagID, cur_tagID, ID_end] += 1.0
                trigrams[cur_tagID, ID_end, ID_start] += 1.0
                prev_tagID, cur_tagID = ID_end, ID_start
        # f.close()

        # add last trigram if file did not end with white line
        if prev_tagID != ID_end:
            trigrams[prev_tagID, cur_tagID, ID_end] += 1.0
            trigrams[cur_tagID, ID_end, ID_start] += 1.0

        return trigrams

    def make_hmm(self, trigrams, emission, tagIDs, wordIDs, smoothing=None):
        """
        Make an HMM object.

        :param trigrams: A 3-dimensional matrix with trigram counts.
        :param emission: A 2-dimensional matrix with lexical counts.
        :param tagIDs:  A map from tags to IDs.
        :type tagIDs:   dictionary
        :param wordIDs: A map from words to IDs.
        :type wordIDs:  dictionary
        :param smoothing:   Optional argument to provide the lambda
                            values for linear interpolation.
        """
        transition_dict = self.get_transition_probs(trigrams, smoothing)
        emission_dict = self.get_emission_probs(emission)
        hmm = HMM2(transition_dict, emission_dict, tagIDs, wordIDs)
        return hmm

    def weighted_lexicon_smoothing(self, lexicon_counts, unlabeled_dict, ratio=1.0):
        """
        Smooth the lexicon by adding frequency counts f(w,tag) for
        all words and tags in the lexicon. The total frequency added
        is a ratio of the number of times a word is already in the
        lexicon. The frequency is then split among all possible tags.
        An exception exists for punctuation tags and words, whose frequency
        counts will remain unchanged.
        """
        nr_of_tags = len(self.tagIDs) - 2

        # check if punctiation tag is in lexicon
        punctID = None
        if 'LET' in self.tagIDs:
            nr_of_tags -= 1
            punctID = self.tagIDs['LET']
        word_sums = lexicon_counts.sum(axis=0)

        # loop over words found in unlabeled file
        for word in unlabeled_dict:
            wordID = self.wordIDs[word]

            # If word is in punctuation, assign punctuation tag
            # and continue
            if word in string.punctuation:
                lexicon_counts[punctID, wordID] += 1
                continue

            # check occurences of word
            word_sum_cur = word_sums[wordID]
            if word_sum_cur == 0:
                word_sum_cur = 1

            # compute additional frequencies for word tag pairs
            extra_freq = unlabeled_dict[word]/word_sum_cur*ratio
            lexicon_counts[:-2, wordID] += extra_freq
            if punctID:
                lexicon_counts[punctID, wordID] -= extra_freq

        return lexicon_counts

    def lexicon_dict_add_unlabeled(self, word_dict, lexicon):
        """
        Add counts to all words in an unlabeled file. It is assumed all
        words are assigned IDs yet and exist in the emission matrix.
        Currently the added counts are equally divided over all input tags,
        and also regardless of how often the word occurred in the unlabeled file.
        Later I should implement a more sophisticated initial estimation,
        and do some kind of scaling to prevent the unlabeled words from becoming
        too influential (or not influential enough).
        """
        # create set with tagIDs
        new_lexicon = copy.copy(lexicon)
        word_IDs, punctuation_IDs = set([]), set([])
        for word in word_dict:
            if word not in string.punctuation:
                word_IDs.add(self.wordIDs[word])
            else:
                punctuation_IDs.add(self.wordIDs[word])
        word_IDs = tuple(word_IDs)
        if 'LET' in self.tagIDs:
            count_per_tag = 1.0/float(lexicon.shape[0]-3)
            punctuation_ID = self.tagIDs['LET']
            new_lexicon[:punctuation_ID, word_IDs] += count_per_tag
            new_lexicon[:punctuation_ID+1:-2, word_IDs] += count_per_tag
            new_lexicon[punctuation_ID, tuple(punctuation_IDs)] += 1.0
        else:
            count_per_tag = 1.0/float(lexicon.shape[0]-2)
            if len(punctuation_IDs) == 0:
                new_lexicon[:-2, word_IDs] += count_per_tag
            else:
                print "No punctuation tag is provided"
                raise KeyError
        return new_lexicon

    def unlabeled_make_word_list(self, input_data):
        """
        Make a dictionary with all words in
        unlabeled file.
        """
        data = self.get_data(input_data, delimiter=True)
        words = Counter(itertools.chain(*data))
        return words

    def labeled_make_word_list(self, input_data):
        """
        Make a dictionary with all words in a
        labeled file.
        """
        data = self.get_data(input_data, delimiter=True)
        word_dict = Counter([item[0] for item in data if item != []])
        return word_dict
        word_dict = {}
        for line in data:
            try:
                word, tag = line.split()
                word_dict[word] = word_dict.get(word, 0) + 1
            except ValueError:
                continue
        return word_dict

    def get_transition_probs(self, trigram_counts, smoothing=None):
        """
        Get trigram probabilities from a frequency matrix.
        :param smoothing:   give a list with lambdas to smooth the probabilities
                            with linear interpolation
        :type smoothing     list
        """

        # Impossible events mess up the probability model, so that the counts
        # do not add up to 1, I should do something about this.

        trigram_sums = trigram_counts.sum(axis=2)
        trigram_sums[trigram_sums == 0.0] = 1.0
        trigram_probs = trigram_counts / trigram_sums[:, :, numpy.newaxis]

        if not smoothing:
            return trigram_probs

        # Check if lambda values sum up to one
        assert sum(smoothing) == 1.0, "lamdba parameters do not add up to 1"

        # smooth counts to keep prob model consisent
        smoothed_counts = trigram_counts + 0.001
        smoothed_counts = self.reset_smoothed_counts(smoothed_counts)
        smoothed_sums = smoothed_counts.sum(axis=2)
        smoothed_sums[smoothed_sums == 0.0] = 1.0
        smoothed_probs = smoothed_counts / smoothed_sums[:, :, numpy.newaxis]

        # compute bigram counts
        # note that this only works if the counts are generated
        # from one file with the generator from this class
        bigram_counts = self.reset_bigram_counts(smoothed_sums)
        bigram_probs = bigram_counts/bigram_counts.sum(axis=1)[:, numpy.newaxis]

        # compute unigram counts
        # note that this only works if the counts are generated
        # from one file with the generator from this class
        unigram_counts = trigram_counts.sum(axis=(0, 2))
        unigram_probs = unigram_counts/unigram_counts.sum()

        # interpolate probabilities
        l1, l2, l3 = smoothing
        smoothed_probs = l1*unigram_probs + l2*bigram_probs + l3*trigram_probs

        # reset probabilites for impossible events
        smoothed_probs = self.reset_smoothed_counts(smoothed_probs)
        
        # normalise again
        sums = smoothed_probs.sum(axis=2)
        sums[sums == 0.0] = 1.0
        probabilities = smoothed_probs/sums[:, :, numpy.newaxis]

        return probabilities

    def reset_bigram_counts(self, bigram_counts):
        """
        Reset counts for impossible bigrams.
        """
        # reset counts for bigrams !### $$$
        bigram_counts[:-2, -2] = 0.0

        # reset counts for bigrams ### !$$$
        bigram_counts[-1, :-2] = 0.0
        bigram_counts[-1, -1] = 0.0

        return bigram_counts

    def reset_smoothed_counts(self, smoothed_counts):
        """
        Reset probabilities for impossible trigrams.
        """
        # reset matrix entries that correspond with trigrams
        # containing TAG $$$, where TAG != ###
        smoothed_counts[:, :-1, -2] = 0.0    # X !### $$$
        smoothed_counts[:-1, -2, :] = 0.0   # !### $$$ X

        # reset matrix entries that correspond with trigrams
        # containing ### TAG where TAG != $$$
        smoothed_counts[:, -1, :-2] = 0.0    # X ### !$$$
        smoothed_counts[:, -1, -1] = 0.0     # X ### ###
        smoothed_counts[-1, :-2, :] = 0.0    # ### !$$$ X
        smoothed_counts[-1, -1, :] = 0.0     # ### ### X

        # smoothed_probs[:, -1, -2] = 1.0     # P($$$| X ###) = 1

        return smoothed_counts

    def transition_dict_add_alpha(self, alpha, trigram_count_matrix):
        """
        Add alpha smoothing for the trigram count dictionary
        """
        # Add alpha to all matrix entries
        trigram_count_matrix += alpha
        # reset matrix entries that correspond with trigrams
        # containing TAG $$$, where TAG != ###
        trigram_count_matrix[:, :-1, -2] = 0.0    # X !### $$$
        trigram_count_matrix[:-1, -2, :] = 0.0    # !### $$$ X
        # reset matrix entries that correspond with trigrams
        # containing ### TAG where TAG != $$$
        trigram_count_matrix[:, -1, :-2] = trigram_count_matrix[:, -1, -1] = 0.0
        trigram_count_matrix[-1, :-2, :] = trigram_count_matrix[-1, -1, :] = 0.0
        return trigram_count_matrix

    def get_emission_probs(self, lexicon):
        """
        Get emission probabilities from a dictionary
        with tag, word counts
        """
        tag_sums = lexicon.sum(axis=1)
        tag_sums[tag_sums == 0.0] = 1
        lexicon /= tag_sums[:, numpy.newaxis]
        return lexicon

    def get_data(self, input_data, delimiter=None):
        """
        If input_data is a filename, return
        a list of the lines in the file with
        this name. Otherwise, return
        inputdata as inputted.
        :rtype: list
        """
        if isinstance(input_data, list):
            data = input_data
        elif isinstance(input_data, str):
            f = open(input_data, 'r')
            data = f.readlines()
            f.close()
        else:
            return ValueError

        if delimiter and not isinstance(data[0], list):
            data = [x.split() for x in data]

        return data
