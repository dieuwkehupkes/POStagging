from HMM2 import *
import copy
from decimal import *
import string

class HMM2_generator:
	"""
	Functions to generate HMMs
	"""
	def __init__(self):
		getcontext.prec = 50

	def init_transition_dict(self, tags):
		"""
		Initialise an empty transition dictionary with all
		possible trigrams.
		"""
		transition_dict = {}
		for tag1 in tags:
			transition_dict[tag1] = {}
			for tag2 in tags:
				transition_dict[tag1][tag2] = {}
				for tag3 in tags:
					transition_dict[tag1][tag2][tag3] = 0
		return transition_dict
	
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
	
	def get_hmm_dicts_from_file(self, input_file, tags=None):
		"""
		Get hmm dictionaries from a file containing words and
		tags separated by a tab. Sentences are delimited by
		newlines.
		Trigrams stop at the end of the sentence, but both the
		beginning and end of a sentence are included in the 
		trigrams.
		"""
		f = open(input_file,'r')
		prev_tag, cur_tag = "###", "$$$"	#beginning of sentence
		trigrams = {}
		if tags:
			trigrams = self.init_transition_dict(tags)
		emission = {}
		for line in f:
			try:
				word, tag = line.split()
				trigrams = self.add_trigram_count(trigrams, prev_tag, cur_tag, tag)
				emission = self.add_word_count(emission, tag, word)
				prev_tag = cur_tag
				cur_tag = tag
			except ValueError:
				#end of sentence
				trigrams = self.add_trigram_count(trigrams, prev_tag, cur_tag, "###")
				prev_tag, cur_tag = "###", "$$$"
		f.close()
		#add last trigram if file did not end with white line
		if prev_tag != "###":
			self.add_trigram_count(trigrams, prev_tag, cur_tag, "###")
		return trigrams, emission
	
	def make_hmm(self, trigrams, emission):
		"""
		Return a HMM object
		"""
		transition_dict = self.get_transition_probs(trigrams)
		emission_dict = self.get_emission_probs(emission)
		hmm = HMM2(transition_dict, emission_dict)
		return hmm
	
	def add_trigram_count(self, counting_dict, tag1, tag2, tag3):
		"""
		Add given trigram to the trigram count. 
		"""
		counting_dict[tag1] = counting_dict.get(tag1, {})
		counting_dict[tag1][tag2] = counting_dict[tag1].get(tag2, {})
		counting_dict[tag1][tag2][tag3] = counting_dict[tag1][tag2].get(tag3, Decimal('0')) + 1
		return counting_dict
	
	def add_word_count(self, word_count_dict, tag, word):
		"""
		Add tag, word count to a counting dictionary.
		"""
		word_count_dict[tag] = word_count_dict.get(tag,{})
		word_count_dict[tag][word] = word_count_dict[tag].get(word,Decimal('0')) + 1
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
			
					

	def transition_dict_add_alpha(self, alpha, trigram_count_dict,tags):
		"""
		Add alpha smoothing for the trigram count dictionary
		"""
		t_dict = copy.copy(trigram_count_dict)
		for tag1 in tags:
			# count for starting a sentence with tag tag1
			t_dict['###']['$$$'][tag1] = t_dict['###']['$$$'].get(tag1,Decimal('0')) + Decimal(str(alpha))
			# create an entry for tag1, if it does not exist yet
			t_dict[tag1] = t_dict.get(tag1,{})
			# create an entry for start, tag1 if it doesn't exist yet
			t_dict['$$$'][tag1] = t_dict['$$$'].get(tag1,{})
			# count for having a one word sentence with tag tag1
			t_dict['$$$'][tag1]['###'] = t_dict['$$$'][tag1].get('$$$',Decimal('0')) + Decimal(str(alpha))
			for tag2 in tags:
				# count for starting a sentence with tag1 tag2
				t_dict['$$$'][tag1][tag2] = t_dict['$$$'][tag1].get(tag2,Decimal('0')) + Decimal(str(alpha))
				# create an entry for trigrams starting with tag1 tag2 if it doesn't exist yet
				t_dict[tag1][tag2] = t_dict[tag1].get(tag2,{})
				# count for sentence ending with tag1 tag2.
				t_dict[tag1][tag2]['###'] = t_dict[tag1][tag2].get('###',Decimal('0')) + Decimal(str(alpha))
				for tag3 in tags:
					# count for trigram tag1 tag2 tag3
					t_dict[tag1][tag2][tag3] = t_dict[tag1][tag2].get(tag3,Decimal('0')) + Decimal(str(alpha))
		return t_dict
	
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
	
	def get_transition_probs(self, trigram_count_dict):
		"""
		Get transition probabilities from a dictionary with
		trigram counts
		"""
		transition_dict = copy.deepcopy(trigram_count_dict)
		for tag1 in trigram_count_dict:
			for tag2 in trigram_count_dict[tag1]:
				total = sum(trigram_count_dict[tag1][tag2].values())
				for tag3 in trigram_count_dict[tag1][tag2]:
					transition_dict[tag1][tag2][tag3] = transition_dict[tag1][tag2][tag3]/total
		return transition_dict
					


if __name__ == '__main__':
	f = sys.argv[1]
	generator = HMM2_generator()
	d1, d2 = generator.get_hmm_dicts_from_file(f)
	tags = ['N','V','###','$$$','LID','VZ']
	d1 = generator.transition_dict_add_alpha(0.1,d1, tags)
	d2 = generator.emission_dict_add_alpha(0.1,d2,set(['een','hond','loopt','naar','huis']))
	hmm = generator.make_hmm(d1, d2)
	sequence = 'een hond loopt naar huis'.split()
	tags = ['LID','V','VZ','N']
	print hmm.compute_probability(sequence, tags)
	
