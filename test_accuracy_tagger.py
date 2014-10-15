"""
Test the accuracy of a syntactic and lexical model derived from a trainingset
on a testset.
"""

from HMMgenerator import HMM2_generator as gen
from HMM2 import HMM2
import sys
from IterativeTraining import Train
from collections import Counter
import pickle
# import numpy


def run(ftrain, feval):
    
    generator = gen()

    data_train = generator.get_data(ftrain, delimiter=True)
    data_train.append([])

    # data_train = open(ftrain, 'r').readlines()
    # data_train.append('\n')
    nr_of_sentences = data_train.count([])
    
    syntactic_smoothing = [0.01, 0.09, 0.9]
    lexical_smoothing = 0.1
    
    T = Train(None)

    tags = generator.find_tags(data_train)
    tagIDs = generator.generate_tag_IDs(tags)
    evaluation = T.load_evaluation(feval)

    words_unlabeled = make_word_list(evaluation)

    tokens, sentences, accuracy_list = [], [], []

    for percentage in [100]:
        l = int(percentage * 0.01 * nr_of_sentences)
        
        # create data to generate syntactic model
        i = -1
        for j in xrange(l):
            i = data_train.index([], i+1)
        training_data = data_train[:i+1]

        nr_of_tokens = len(training_data) - l

        print 'number of training sentences:', l
        print 'number of training tokens:', nr_of_tokens

        # tagIDs, transition_matrix = pickle.load(open('lassy.train.syntactic_model', 'rb'))

        # generate trigrams
        # print "Generate syntactic model"
        trigram_counts = generator.get_trigrams_from_file(training_data, tagIDs)
        transition_matrix = generator.get_transition_probs(trigram_counts, smoothing=syntactic_smoothing)
        # pickle.dump([tagIDs, transition_matrix], open('lassy.train.syntactic_model', 'wb'))

        words_labeled = generator.labeled_make_word_list(training_data)

        # print "Generate lexical model"
        all_words = set(words_labeled.keys()).union(set(words_unlabeled.keys()))
        wordIDs = generator.generate_lexicon_IDs(all_words)
        lex_dict = generator.get_lexicon_counts(training_data, tagIDs, wordIDs)
        lex_dict_smoothed = generator.weighted_lexicon_smoothing(lex_dict, words_unlabeled, ratio=lexical_smoothing)
        lexicon = generator.get_emission_probs(lex_dict_smoothed)
        hmm = HMM2(transition_matrix, lexicon, tagIDs, wordIDs)

        return
    
        # print "Tag corpus and evaluate accuracy"
        # accuracy = T.compute_accuracy(hmm, evaluation, ignore_tags=set(['LET']))
        accuracy = T.compute_accuracy(hmm, evaluation)
        print '\n', accuracy

        tokens.append(nr_of_tokens)
        sentences.append(l)
        accuracy_list.append(accuracy)

    print tokens
    print sentences
    print accuracy_list


def make_word_list(evaluation_set):
    all_words = []
    for item in evaluation_set:
        all_words += item[0]
    words = Counter(all_words)
    return words

if __name__ == '__main__':
    ftrain, feval = sys.argv[1], sys.argv[2]
    run(ftrain, feval)
