"""
Functions to generate HMMs.
Add scaled smoothing lexicon dict (based on how often a word occured)
"""

from HMM2 import HMM2
import string
import numpy
import copy


class HMM2_generator:
    """
    Initialise an HMM generator.
    """
    def __init__(self):
        pass

    def init_transition_matrix(self, tags):
        """
        Transitions are stored in a 3-dimensional matrix.
        Initialising an empty transition matrix thus
        equals generating an empty matrix of size N*N*N,
        where N is the number of tags.
        """
        all_tags = set(tags).union(set(['$$$', '###']))
        N = len(all_tags)
        transition_matrix = numpy.zeros(shape=(N, N, N), dtype=numpy.float64)
        return transition_matrix

    def init_lexicon_matrix(self, words, nr_of_tags):
        """
        Initialise an empty lexicon matrix.
        """
        nr_of_words = len(words)
        lexicon_matrix = numpy.zeros(shape=(nr_of_tags+2, nr_of_words), dtype=numpy.float64)
        return lexicon_matrix

    def get_words_from_file(self, input_file):
        """
        Get all words from a file containing
        tokenised sentences.
        """
        f = open(input_file, 'r')
        words = set([])
        for line in f:
            try:
                words = words.union(set(line.split()))
            except IndexError:
                continue
        return words

    def generate_tag_IDs(self, tags):
        """
        Generate a dictionary that stores the relation
        between tags and transition/emission matrix.
        The ID generated for a tag represents the index
        under which the tag is stored in these matrices.
        """
        self.tagIDs = {}
        i = 0
        for tag in tags:
            self.tagIDs[tag] = i
            i += 1
        self.tagIDs['$$$'] = i
        self.tagIDs['###'] = i+1
        return self.tagIDs

    def generate_lexicon_IDs(self, words):
        """
        Generate a dictionary that stores the relation between
        words and emission matrix. The ID generated for a
        word is the index that can be used to look up the
        word in the emission matrix
        """
        self.wordIDs = {}
        i = 0
        for word in words:
            self.wordIDs[word] = i
            i += 1
        return self.wordIDs

    def get_hmm_dicts_from_file(self, input_file, tags, words):
        """
        Generate hmm matrices from a file containing lines
        with words and tags separated by a tab. Sentences are delimited by
        newlines.
        Trigrams stop at the end of the sentence, but both the
        beginning and end of a sentence are included in the
        trigrams.
        """
        f = open(input_file, 'r')
        trigrams = self.init_transition_matrix(tags)
        emission = self.init_lexicon_matrix(words, len(tags))
        wordIDs = self.generate_lexicon_IDs(words)
        tagIDs = self.generate_tag_IDs(tags)
        ID_end, ID_start = tagIDs['###'], tagIDs['$$$']
        prev_tagID, cur_tagID = ID_end, ID_start         # beginning of sentence
        for line in f:
            try:
                word, tag = line.split()
                wordID, tagID = wordIDs[word], tagIDs[tag]
                trigrams[prev_tagID, cur_tagID, tagID] += 1.0
                emission[tagID, wordID] += 1.0
                prev_tagID = cur_tagID
                cur_tagID = tagID
            except ValueError:
                # end of sentence
                trigrams[prev_tagID, cur_tagID, ID_end] += 1.0
                prev_tagID, cur_tagID = ID_end, ID_start
        f.close()

        # add last trigram if file did not end with white line
        if prev_tagID != ID_end:
            trigrams[prev_tagID, cur_tagID, ID_end] += 1.0
        return trigrams, emission

    def make_hmm(self, trigrams, emission):
        """
        Return a HMM object
        """
        transition_dict = self.get_transition_probs(trigrams)
        emission_dict = self.get_emission_probs(emission)
        hmm = HMM2(transition_dict, emission_dict, self.tagIDs, self.wordIDs)
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

        for word in unlabeled_dict:
            wordID = self.wordIDs[word]
            if word in string.punctuation:
                lexicon_counts[punctID, wordID] += 1
                continue

            word_sum_cur = word_sums[wordID]
            if word_sum_cur == 0.0:
                word_sum_cur = 1.0

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

    def unlabeled_make_word_list(self, unlabeled_file):
        """
        Make a dictionary with all words in
        unlabeled file.
        """
        f = open(unlabeled_file, 'r')
        word_dict = {}
        for line in f:
            words = line.split()
            for word in words:
                word_dict[word] = word_dict.get(word, 0.0) + 1.0
        f.close()
        return word_dict

    def labeled_make_word_list(self, labeled_file):
        """
        Make a dictionary with all words in a
        labeled file.
        """
        f = open(labeled_file, 'r')
        word_dict = {}
        for line in f:
            try:
                word, tag = line.split()
                word_dict[word] = word_dict.get(word, 0) + 1
            except ValueError:
                continue
        f.close()
        return word_dict

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

    def get_transition_probs(self, trigram_count_matrix):
        """
        Get transition probabilities from a dictionary with
        trigram counts
        """
        # compute the sums for every row
        tag_sums = trigram_count_matrix.sum(axis=2)
        tag_sums[tag_sums == 0.0] = 1.0
        # divide the transition matrix by the broadcasted tag sums
        trigram_count_matrix /= tag_sums[:, :, numpy.newaxis]
        return trigram_count_matrix
