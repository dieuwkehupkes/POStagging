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
	
	def get_hmm_dicts_from_file(self, input_file, tags):
		"""
		Generate hmm matrices from a file containing lines
		with words and tags separated by a tab. Sentences are delimited by
		newlines.
		Trigrams stop at the end of the sentence, but both the
		beginning and end of a sentence are included in the 
		trigrams.
		"""
		f = open(input_file,'r')
		prev_tag, cur_tag = "###", "$$$"	#beginning of sentence
		trigrams = self.init_transition_matrix(tags)
		IDs = self.generate_tag_IDs(tags)
		emission = {}
		for line in f:
			try:
				word, tag = line.split()
				trigrams[IDs[prev_tag],IDs[cur_tag], IDs[tag]] += Decimal('1.0')
				emission = self.add_word_count(emission, tag, word)
				prev_tag = cur_tag
				cur_tag = tag
			except ValueError:
				#end of sentence
				trigrams[IDs[prev_tag],IDs[cur_tag], IDs['###']] += Decimal('1.0')
				prev_tag, cur_tag = "###", "$$$"
		f.close()
		#add last trigram if file did not end with white line
		if prev_tag != "###":
			trigrams[IDs[prev_tag], IDs[cur_tag], IDs["###"]] += Decimal('1.0')
		return trigrams, emission
	
	def make_hmm(self, trigrams, emission):
		"""
		Return a HMM object
		"""
		transition_dict = self.get_transition_probs(trigrams)
		emission_dict = self.get_emission_probs(emission)
		tagIDs = self.tagIDs
		hmm = HMM2(transition_dict, emission_dict,self.tagIDs)
		return hmm
	
	def add_word_count(self, word_count_dict, tag, word):
		"""
		Add tag, word count to a counting dictionary.
		"""
		word_count_dict[tag] = word_count_dict.get(tag,{})
		word_count_dict[tag][word] = word_count_dict[tag].get(word,Decimal('0')) + Decimal('1.0')
		return word_count_dict

	def lexicon_dict_add_unlabeled(self, word_dict, lexicon_dict, tags):
		"""
		For every word in an unlabeled file, add counts to a dictionary
		with lexicon counts. The counts are equally diveded over all inputted tags,
		later I could maybe implement something with more sophisticated
		initial estimations.
		"""
		words = word_dict
		i = 0
		l_dict = copy.copy(lexicon_dict)
		count_per_tag = Decimal('1')/Decimal(len(tags))
		#check whether every tag exists in lexicon
		for tag in tags:
			l_dict[tag] = l_dict.get(tag,{})
		for word in words:
			if word in string.punctuation:
				l_dict['LET'][word] = l_dict['LET'].get(word, Decimal('0')) + Decimal('1') 
				continue
			for tag in tags.difference(['LET']):
				#l_dict[tag][word] = l_dict[tag].get(word,Decimal('0')) + Decimal(words[word])*count_per_tag
				l_dict[tag][word] = l_dict[tag].get(word,Decimal('0')) + Decimal('1')
			i += 1
		return l_dict
	
	def make_word_list(self, unlabeled_file):
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
	
	def get_emission_probs(self, word_count_dict):
		"""
		Get emission probabilities from a dictionary
		with tag, word counts
		"""
		emission_dict = copy.deepcopy(word_count_dict)
		emission_dict["$$$"] = {'$$$':1}
		emission_dict["###"] = {'###':1}
		for tag in word_count_dict:
			total = sum(word_count_dict[tag].values())
			for word in word_count_dict[tag]:
				emission_dict[tag][word] = word_count_dict[tag][word]/total
				emission_dict["$$$"][word] = 0
				emission_dict["###"][word] = 0
		return emission_dict	
	
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
	tags = set(['N','V','LID','VZ'])
	d1, d2 = generator.get_hmm_dicts_from_file(f, tags)
	d1 = generator.transition_dict_add_alpha(0.1, d1)
	words = set(['een','hond','loopt','naar','huis'])
	d2 = generator.lexicon_dict_add_unlabeled(words, d2, tags)
	hmm = generator.make_hmm(d1, d2)
	#hmm.print_trigrams()
	sequence = 'een hond loopt naar huis'.split()
	tags = ['LID','N','V','VZ','N']
	print hmm.compute_probability(sequence, tags)
	
