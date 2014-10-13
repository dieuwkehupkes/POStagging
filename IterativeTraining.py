"""
A class for iterative semi-supervised training.
explain better
MOVE COMPUTE EXPECTED COUNTS FUNCTION TO HMM2 CLASS
"""

# from HMMgenerator import HMM2_generator as gen
from HMM2 import HMM2
from ForwardBackward import ForwardBackward
from HMMgenerator import HMM2_generator as gen
from Viterbi import Viterbi
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

    def train(self, start, iterations, scaling):
        """
        Train the model using the labeled and unlabeled corpus.
        Some more explanation.
        """
        hmm = start
        lexicon_basis = scaling * self.D.lex_dict
        for i in xrange(iterations):
            print "iteration", i
            hmm = self.iteration(hmm, lexicon_basis)
        return hmm

    def train_and_test(self, start, iterations, scaling, evaluation_set):
        hmm = start
        evaluation = self.load_evaluation(evaluation_set)
        lexicon_basis = scaling * self.lex_basis
        for i in xrange(iterations):
            print "iteration", i
            accuracy = self.compute_accuracy(hmm, evaluation)
            print "accuracy before iteration: %f" % (accuracy)
            hmm = self.iteration(hmm, lexicon_basis)
            raw_input()

        print "accuracy after training:", accuracy
        return hmm

    def iteration(self, hmm, labeled_counts):
        """
        Do one iteration of training.
        """
        f = open(self.D.unlabeled, 'r')
        sum_expected_counts = numpy.zeros(shape=hmm.emission.shape, dtype=numpy.float64)
        for line in f:
            training = ForwardBackward(line, hmm)
            expected_counts = training.compute_expected_counts()
            sum_expected_counts = training.update_lexical_dict(expected_counts, sum_expected_counts)
        new_lexicon = training.update_lexical_dict(sum_expected_counts, labeled_counts)
        f.close()
        new_hmm = HMM2(hmm.transition, new_lexicon, hmm.tagIDs, hmm.wordIDs)
        return new_hmm

    def load_evaluation(self, evaluation):
        """
        Load an evaluation file in memory by creating a
        dictionary with sentences as keys and their correct
        tag sequence as values.
        The evaluation file should contain lines with a
        word and a tag separated by a tab. Sentences are
        delimited by newlines.
        """
        f = open(evaluation, 'r')
        evaluation = {}
        sentence = ''
        tag_sequence = []
        for line in f:
            try:
                word, tag = line.split()
                sentence = sentence + ' ' + word
                tag_sequence.append(tag)
            except ValueError:
                if sentence != '':
                    evaluation[sentence] = tag_sequence
                    sentence = ''
                    tag_sequence = []

        f.close()

        if sentence != '':
            evaluation[sentence] = tag_sequence
        return evaluation

    def compute_accuracy(self, hmm, evaluation, ignore_tags=set([])):
        """
        Compute the accuracy of hmm tags on an
        validation dictionary.
        """
        V = Viterbi(hmm)
        accuracy = 0.0
        print len(evaluation)
        i = 0
        for sentence in evaluation:
            i += 1
            print i
            hmm_tags = V.compute_best_parse(sentence)[1]
            validation_tags = evaluation[sentence]
            accuracy += self.accuracy(hmm_tags, validation_tags, ignore_tags)
        total_accuracy = accuracy/len(evaluation)
        return total_accuracy

    def accuracy(self, hmm_tags, validation_tags, ignore_tags):
        """
        Compute the accuracy of the hmm-assigned tags.
        """
        accuracy = 0.0
        l = 0
        try:
            for i in xrange(len(hmm_tags)):
                if hmm_tags[i] == validation_tags[i] and validation_tags[i] not in ignore_tags:
                    accuracy += 1.0
                    l += 1
            accuracy_sentence = accuracy/l
        except IndexError:
            print hmm_tags, validation_tags
            accuracy_sentence = 0
        return accuracy_sentence
