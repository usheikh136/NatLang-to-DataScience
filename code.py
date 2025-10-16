import json
from collections import Counter
import numpy as np
import nltk
from nltk.data import find
import gensim
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
nltk.download('word2vec_sample')

########-------------- PART 1: LANGUAGE MODELING --------------########

class NgramLM:
	def __init__(self):
		"""N-gram Language Model"""
		# Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram = {}

		# Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram_weights = {}

	def load_trigrams(self):
		"""
		Loads the trigrams from the data file and fills the dictionaries defined above.

		Parameters
		----------

		Returns
		-------
		"""
		with open("/content/drive/My Drive/data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
			lines = f.readlines()
			for line in lines:
				word1, word2, word3, count = line.strip().split()
				if (word1, word2) not in self.bigram_prefix_to_trigram:
					self.bigram_prefix_to_trigram[(word1, word2)] = []
					self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
				self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
				self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

	def top_next_word(self, word1, word2, n=10):
			next_words = []
			probs = []

			bigram = (word1, word2)
			if bigram in self.bigram_prefix_to_trigram:
					nw = self.bigram_prefix_to_trigram[bigram]
					nw_counts = self.bigram_prefix_to_trigram_weights[bigram]
					count_total = sum(nw_counts)

					# Sort the next words by their counts in descending order and keep the top n words
					nw_sorted = sorted(range(len(nw)), key=lambda i: nw_counts[i], reverse=True)
					top_n = nw_sorted[:n]

					# Calculate probabilities and populate next_words and probs lists
					for i in top_n:
							w = nw[i]
							prob = nw_counts[i] / count_total

							next_words.append(w)
							probs.append(prob)

			return next_words, probs
	
	def sample_next_word(self, word1, word2, n=10):
			"""
			Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

			Parameters
			----------
			word1: str
				The first word in the bigram.
			word2: str
				The second word in the bigram.
			n: int
				Number of words to return.
				
			Returns
			-------
			next_words: list
				The sampled n next words.
			probs: list
				The probabilities corresponding to the retrieved words.
			"""
			next_words = []
			probs = []

			bigram = (word1, word2)
			if bigram in self.bigram_prefix_to_trigram:
					nw = self.bigram_prefix_to_trigram[bigram]
					nw_counts = self.bigram_prefix_to_trigram_weights[bigram]
					count_total = sum(nw_counts)

					nw_probs = []
					for count in nw_counts:
							nw_probs.append(count / count_total)
			
					samples = np.random.choice(nw, size=n, replace=False, p=nw_probs)

					# Calculate probabilities and populate next_words and probs lists
					for s in samples:
							next_words.append(s)
							probs.append(nw_probs[nw.index(s)])

			return next_words, probs
	
	def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
			"""
			Generate sentences using beam search.

			Parameters
			----------
			prefix: str
				String containing two (or more) words separated by spaces.
			beam: int
				The beam size.
			sampler: Callable
				The function used to sample next word.
			max_len: int
				Maximum length of sentence (as measure by number of words) to generate (excluding "<EOS>").
				
			Returns
			-------
			sentences: list
				The top generated sentences
			probs: list
				The probabilities corresponding to the generated sentences
			"""
			sentences = []
			probs = []

			return sentences, probs

#####------------- CODE TO TEST YOUR FUNCTIONS FOR PART 1

# Define your language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()

print("------------- Evaluating top next word prediction -------------")
next_words, probs = language_model.top_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)
# Your first 5 lines of output should be exactly:
# a 0.807981220657277
# the 0.06948356807511737
# pandemic 0.023943661971830985
# this 0.016901408450704224
# an 0.0107981220657277

print("------------- Evaluating sample next word prediction -------------")
next_words, probs = language_model.sample_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)
# My first 5 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# a 0.807981220657277
# pandemic 0.023943661971830985
# august 0.0018779342723004694
# stage 0.0018779342723004694
# an 0.0107981220657277

print("------------- Evaluating beam search -------------")
sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> biden calls for a 30 bonus URL #cashgem #cashappfriday #stayathome <EOS> 0.0002495268686322749
# <BOS1> <BOS2> biden says all u.s. governors should mandate masks <EOS> 1.6894510541025754e-05
# <BOS1> <BOS2> biden says all u.s. governors question cost of a pandemic <EOS> 8.777606198953028e-07

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
# My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# <BOS1> <BOS2> biden is elected <EOS> 0.001236227651321991
# <BOS1> <BOS2> biden dropping ten points given trump a confidence trickster URL <EOS> 5.1049579351466146e-05
# <BOS1> <BOS2> biden dropping ten points given trump four years <EOS> 4.367575122292103e-05




########-------------- PART 2: Semantic Parsing --------------########

class SemanticParser:
	def __init__(self):
		"""
		Basic Semantic Parser
		"""
		self.parser_files = "data/semantic-parser"
		self.train_data = []
		self.test_questions = []
		self.test_answers = []
		self.intents = set()
		self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)
		self.classifier = LogisticRegression(random_state=42)

		# Let's stick to one target intent.
		self.target_intent = "AddToPlaylist"
		self.target_intent_slot_names = set()
		self.target_intent_questions = []

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		with open(f'{self.parser_files}/train_questions_answers.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.train_data.append(json.loads(line))

		with open(f'{self.parser_files}/val_questions.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_questions.append(json.loads(line))

		with open(f'{self.parser_files}/val_answers.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_answers.append(json.loads(line))

		for example in self.train_data:
			self.intents.add(example['intent'])

	def predict_intent_using_keywords(self, question):
		"""
		Predicts the intent of the question using custom-defined keywords.

		Parameters
		----------
		question: str
			The question whose intent is to be predicted.
			
		Returns
		-------
		intent: str
			The predicted intent.
		"""
		intent = ""
		# Fill in your code here.
		pass
		return intent

	def evaluate_intent_accuracy(self, prediction_function_name):
		"""
		Gives intent wise accuracy of your model.

		Parameters
		----------
		prediction_function_name: Callable
			The function used for predicting intents.
			
		Returns
		-------
		accs: dict
			The accuracies of predicting each intent.
		"""
		correct = Counter()
		total = Counter()
		for i in range(len(self.test_questions)):
			q = self.test_questions[i]
			gold_intent = self.test_answers[i]['intent']
			if prediction_function_name(q) == gold_intent:
				correct[gold_intent] += 1
			total[gold_intent] += 1
		accs = {}
		for intent in self.intents:
			accs[intent] = (correct[intent]/total[intent])*100
		return accs

	def get_sentence_representation(self, sentence):
		"""
		Gives the average word2vec representation of a sentence.

		Parameters
		----------
		sentence: str
			The sentence whose representation is to be returned.
			
		Returns
		-------
		sentence_vector: np.ndarray
			The representation of the sentence.
		"""
		# Fill in your code here
		sentence_vector = np.zeros(300)
		pass
		return sentence_vector
	
	def train_logistic_regression_intent_classifier(self):
		"""
		Trains the logistic regression classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		# Fill in your code here
		pass
	
	def predict_intent_using_logistic_regression(self, question):
		"""
		Predicts the intent of the question using the logistic regression classifier.

		Parameters
		----------
		question: str
			The question whose intent is to be predicted.
			
		Returns
		-------
		intent: str
			The predicted intent.
		"""
		# Fill in your code here
		intent = ""
		pass
		return intent
	
	def get_target_intent_slots(self):
		"""
		Get the slots for the target intent.

		Parameters
		----------
			
		Returns
		-------
		"""
		for sample in self.train_data:
			if sample['intent'] == self.target_intent:
				for slot_name in sample['slots']:
					self.target_intent_slot_names.add(slot_name)

		for i, question in enumerate(self.test_questions):
			if self.test_answers[i]['intent'] == self.target_intent:
				self.target_intent_questions.append(question)
	
	def predict_slot_values(self, question):
		"""
		Predicts the values for the slots of the target intent.

		Parameters
		----------
		question: str
			The question for which the slots are to be predicted.
			
		Returns
		-------
		slots: dict
			The predicted slots.
		"""
		words = question.split()
		slots = {}
		for slot_name in self.target_intent_slot_names:
			slots[slot_name] = None
		for slot_name in self.target_intent_slot_names:
			# Fill in your code to idenfity the slot value. By default, they are initialized to None.
			pass
		return slots
	
	def get_confusion_matrix(self, slot_prediction_function, questions, answers):
		"""
		Find the true positive, true negative, false positive, and false negative examples with respect to the prediction 
		of a slot being active or not (irrespective of value assigned).

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
		questions: list
			The test questions
		answers: list
			The ground-truth test answers
			
		Returns
		-------
		tp: dict
			The indices of true positive examples are listed for each slot
		fp: dict
			The indices of false positive examples are listed for each slot
		tn: dict
			The indices of true negative examples are listed for each slot
		fn: dict
			The indices of false negative examples are listed for each slot
		"""
		tp = {}
		fp = {}
		tn = {}
		fn = {}
		for slot_name in self.target_intent_slot_names:
			tp[slot_name] = []
		for slot_name in self.target_intent_slot_names:
			fp[slot_name] = []
		for slot_name in self.target_intent_slot_names:
			tn[slot_name] = []
		for slot_name in self.target_intent_slot_names:
			fn[slot_name] = []
		for i, question in enumerate(questions):
			# Fill in your code here
			pass
		return tp, fp, tn, fn
	
	def evaluate_slot_prediction_recall(self, slot_prediction_function):
		"""
		Evaluates the recall for the slot predictor. Note: This also takes into account the exact value predicted for the slot 
		and not just whether the slot is active like in the get_confusion_matrix() method

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
			
		Returns
		-------
		accs: dict
			The recall for predicting the value for each slot.
		"""
		correct = Counter()
		total = Counter()
		# predict slots for each question
		for i, question in enumerate(self.target_intent_questions):
			i = self.test_questions.index(question) # This line is added after the assignment release
			gold_slots = self.test_answers[i]['slots']
			predicted_slots = slot_prediction_function(question)
			for name in self.target_intent_slot_names:
				if name in gold_slots:
					total[name] += 1.0
					if predicted_slots.get(name, None) != None and predicted_slots.get(name).lower() == gold_slots.get(name).lower(): # This line is updated after the assignment release
						correct[name] += 1.0
		accs = {}
		for name in self.target_intent_slot_names:
			accs[name] = (correct[name] / total[name]) * 100
		return accs

#####------------- CODE TO TEST YOUR FUNCTIONS

# Define your semantic parser object
semantic_parser = SemanticParser()
# Load semantic parser data
semantic_parser.load_data()

# Evaluating the keyword-based intent classifier. 
# In our implementation, a simple keyword based classifier has achieved an accuracy of greater than 65 for each intent
print("------------- Evaluating keyword-based intent classifier -------------")
accs = semantic_parser.evaluate_intent_accuracy(semantic_parser.predict_intent_using_keywords)
for intent in accs:
	print(intent + ": " + str(accs[intent]))

# Evaluate the logistic regression intent classifier
# Your intent classifier performance will be 100 if you have done a good job.
print("------------- Evaluating logistic regression intent classifier -------------")
semantic_parser.train_logistic_regression_intent_classifier()
accs = semantic_parser.evaluate_intent_accuracy(semantic_parser.predict_intent_using_logistic_regression)
for intent in accs:
	print(intent + ": " + str(accs[intent]))

# Look at the slots of the target intent
print("------------- Target intent slots -------------")
semantic_parser.get_target_intent_slots()
print(semantic_parser.target_intent_slot_names)

# Evaluate slot predictor
# Our reference implementation got these numbers on the validation set. You can ask others on Slack what they got.
# playlist_owner: 100.0
# music_item: 100.0
# entity_name: 16.666666666666664
# artist: 14.285714285714285
# playlist: 52.94117647058824
print("------------- Evaluating slot predictor -------------")
accs = semantic_parser.evaluate_slot_prediction_recall(semantic_parser.predict_slot_values)
for slot in accs:
	print(slot + ": " + str(accs[slot]))

# Evaluate Confusion matrix examples
print("------------- Confusion matrix examples -------------")
tp, fp, tn, fn = semantic_parser.get_confusion_matrix(semantic_parser.predict_slot_values, semantic_parser.test_questions, semantic_parser.test_answers)
print(tp)
print(fp)
print(tn)
print(fn)