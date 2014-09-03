from HMM2 import *

class HMM2_generator:
	"""
	Functions to generate HMMs
	"""
	def __init__(self):
		#currently it seems nothing needs to be done here
		pass

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
			
			
	
	def get_hmm_from_file(self, input_file, tags=None):
		"""
		Get an HMM from a file containing words and
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
		counting_dict[tag1][tag2][tag3] = counting_dict[tag1][tag2].get(tag3, 0) + 1
		return counting_dict
	
	def add_word_count(self, word_count_dict, tag, word):
		"""
		Add tag, word count to a counting dictionary.
		"""
		word_count_dict[tag] = word_count_dict.get(tag,{})
		word_count_dict[tag][word] = word_count_dict[tag].get(word,0) +1
		return word_count_dict

	def get_emission_probs(self, word_count_dict):
		"""
		Get emission probabilities from a dictionary
		with tag, word counts
		"""
		emission_dict = copy.deepcopy(word_count_dict)
		for tag in word_count_dict:
			total = sum(word_count_dict[tag].values())
			for word in word_count_dict[tag]:
				emission_dict[tag][word] = float(word_count_dict[tag][word])/total
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
					transition_dict[tag1][tag2][tag3] = float(transition_dict[tag1][tag2][tag3])/total
		return transition_dict
					


if __name__ == '__main__':
	f = sys.argv[1]
	generator = HMM2_generator()
	hmm = generator.get_hmm_from_file(f)
	print hmm.compute_probability([('ik','N'), ('loop','V'), ('snel', 'ADV')])
	
