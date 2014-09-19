from HMMgenerator import HMM2_generator as gen
import pickle

tags = set(['LET','LID', 'VZ', 'WW','TSW','ADJ','N','VG','BW','TW','SPEC(e)', 'VNW','VZ+LID','WW+VNW']) 

generator = gen()
words_labeled = generator.labeled_make_word_list('../../Data/StatenvertalingParallel/Test/test.1637.tags.gold')
words_unlabeled =set('inden beginne schiep godt den hemel , ende de aerde .'.split())
all_words = words_unlabeled.union(set(words_labeled.keys()))

trans_dict, lex_dict = generator.get_hmm_dicts_from_file('../../Data/StatenvertalingParallel/Test/test.1637.tags.gold', tags, all_words )

#smooth transition and lexicon matrices
tran_dict = generator.transition_dict_add_alpha(0.5, trans_dict)
lex_dict = generator.lexicon_dict_add_unlabeled(words_unlabeled, lex_dict)
hmm = generator.make_hmm(trans_dict, lex_dict)

pickle.dump(hmm, open('statenvertaling.hmm.pickle', 'w'))
