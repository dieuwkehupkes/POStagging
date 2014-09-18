"""
Functions to generate HMMs.
"""
from HMM2 import *
import copy
from decimal import *
import string

class HMM2_generator:
	"""
	Initialise an HMM generator.
	"""
	def __init__(self, precision = 50):
	 	getcontext().prec = 50
	       

	def init_transition_matrix(self, tags):
		"""
		Transitions are stored in a 3-dimensional matrix.
		Initialising an empty transition matrix thus
		equals generating an empty matrix of size N*N*N,
		where N is the number of tags.
		"""
		all_tags = set(tags).union(set(['$$$','###']))
		N = len(all_tags)
		transition_matrix = numpy.zeros(shape=(N,N,N), dtype=Decimal)
		transition_matrix += Decimal('0.0') 
		return transition_matrix
	
	def init_lexicon_matrix(self, words, nr_of_tags):
		"""
		Initialise an empty lexicon matrix.
		"""
		nr_of_words = len(words)
		lexicon_matrix = numpy.zeros(shape=(nr_of_tags+2, nr_of_words), dtype=Decimal)
		lexicon_matrix += Decimal('0.0')
		return lexicon_matrix
	
	def get_words_from_file(self, input_file):
		"""
		Get all words from a file containing
		tokenised sentences.
		"""
		f = open(input_file,'r')
		words = set([])
		for line in f:
			try:
				words = words.union(set(line.split()))
			except IndexError:
				continue
		return words

	def generate_tag_IDs(self,tags):
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
			i +=1
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
		f = open(input_file,'r')
		trigrams = self.init_transition_matrix(tags)
		emission = self.init_lexicon_matrix(words, len(tags))
		wordIDs = self.generate_lexicon_IDs(words)
		tagIDs = self.generate_tag_IDs(tags)
		ID_end, ID_start = tagIDs['###'], tagIDs['$$$']
		prev_tagID, cur_tagID = ID_end, ID_start	#beginning of sentence
		for line in f:
			try:
				word, tag = line.split()
				wordID, tagID = wordIDs[word], tagIDs[tag]
				trigrams[prev_tagID,cur_tagID, tagID] += Decimal('1.0')
				emission[tagID, wordID] += Decimal('1.0')
				prev_tagID = cur_tagID
				cur_tagID = tagID
			except ValueError:
				#end of sentence
				trigrams[prev_tagID,cur_tagID, ID_end] += Decimal('1.0')
				prev_tagID, cur_tagID = ID_end, ID_start
		f.close()
		#add last trigram if file did not end with white line
		if prev_tagID != ID_end: 
			trigrams[prev_tagID, cur_tagID, ID_end] += Decimal('1.0')
		return trigrams, emission
	
	def make_hmm(self, trigrams, emission):
		"""
		Return a HMM object
		"""
		transition_dict = self.get_transition_probs(trigrams)
		emission_dict = self.get_emission_probs(emission)
		hmm = HMM2(transition_dict, emission_dict,self.tagIDs, self.wordIDs)
		return hmm
	
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
		words = word_dict
		i = 0
		#create set with tagIDs
		word_IDs, punctuation_IDs = set([]), set([])
		for word in word_dict:
			if word not in string.punctuation:
				word_IDs.add(self.wordIDs[word])
			else:
				punctuation_IDs.add(self.wordIDs[word])
		word_IDs = tuple(word_IDs)
		if 'LET' in self.tagIDs:
			count_per_tag = Decimal('1')/Decimal(lexicon.shape[0]-3)
			punctuation_ID = self.tagIDs['LET'] 
			lexicon[:punctuation_ID,word_IDs] += count_per_tag
			lexicon[:punctuation_ID+1:-2, word_IDs] += count_per_tag
			lexicon[punctuation_ID, tuple(punctuation_IDs)] += Decimal('1.0')
		else:
			count_per_tag = Decimal('1')/Decimal(lexicon.shape[0]-2)
			if len(punctuation_IDs) == 0:
				lexicon[:-2,word_IDs] += count_per_tag
			else:
				print "No punctuation tag is provided"
				raise KeyError
		return lexicon
	
	def unlabeled_make_word_list(self, unlabeled_file):
		"""
		Make a dictionary with all words in
		unlabeled file.
		"""
		f = open(unlabeled_file,'r')
		word_dict = {}
		for line in f:
			words = line.split()
			for word in words:
				word_dict[word] = word_dict.get(word,0) + 1
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
				word_dict[word] = word_dict.get(word,0) +1
			except ValueError:
				continue
		f.close()
		return word_dict
			
	def transition_dict_add_alpha(self, alpha, trigram_count_matrix):
		"""
		Add alpha smoothing for the trigram count dictionary
		"""
		#Add alpha to all matrix entries
		trigram_count_matrix += Decimal(str(alpha))
		#reset matrix entries that correspond with trigrams
		#containing TAG $$$, where TAG != ###
		trigram_count_matrix[:,:-1,-2] = Decimal('0')	# X !### $$$
		trigram_count_matrix[:-1,-2,:] = Decimal('0')	# !### $$$ X
		#reset matrix entries that correspond with trigrams
		#containing ### TAG where TAG != $$$
		trigram_count_matrix[:,-1,:-2] = trigram_count_matrix[:,-1,-1] = Decimal('0')
		trigram_count_matrix[-1,:-2,:] = trigram_count_matrix[-1,-1,:] = Decimal('0')
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
		#compute the sums for every row
		tag_sums = trigram_count_matrix.sum(axis=2)
		tag_sums[tag_sums == 0.0] = Decimal('1.0')
		#divide the transition matrix by the broadcasted tag sums
		trigram_count_matrix /= tag_sums[:,:,numpy.newaxis]
		return trigram_count_matrix
					


if __name__ == '__main__':
	f = sys.argv[1]
	generator = HMM2_generator()
	words_labeled = generator.labeled_make_word_list(f)
	words_unlabeled = set(['een','hond','loopt','naar','huis'])
	words = set(words_labeled.keys()).union(words_unlabeled)
	tags = set(['N','V','LID','VZ'])
	d1, d2 = generator.get_hmm_dicts_from_file(f, tags, words)
	d1 = generator.transition_dict_add_alpha(0.1, d1)
	d2 = generator.lexicon_dict_add_unlabeled(words, d2)
	hmm = generator.make_hmm(d1, d2)
	#hmm.print_trigrams()
	sequence = 'een hond loopt naar huis'.split()
	tags = ['LID','N','V','VZ','N']
	print hmm.compute_probability(sequence, tags)
	
