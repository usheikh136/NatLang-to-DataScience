from gensim.models import KeyedVectors
import numpy as np
from sklearn.utils.extmath import randomized_svd
import regex as re
import collections
import math
from numpy.linalg import norm

def top_k_unigrams(tweets, stop_words, k):
  frequencies = []
  for t in tweets:
    words = t.split()
    for word in words:
      if word in stop_words or re.match(r'^[^a-z#]', word):
        continue
      else:
        frequencies.append(word)
  if k == -1:
    return dict(collections.Counter(frequencies))
  else:
    return dict(collections.Counter(frequencies).most_common(k))

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
  pairs = []
  for t in tweets:
    words = t.split()
    for i, word in enumerate(words):
      window = words[max(0,i-context_size):i]+words[i+1:min(len(words),i+1+context_size)]
      for w in window:
        if w in frequent_unigrams:
          pairs.append((word, w))

  return dict(collections.Counter(pairs))


def pmi(word1, word2, unigram_counter, context_counter):
    total = sum(unigram_counter.values())
    
    word1_prob, word2_prob, pair_prob = 1/total,1/total,1/total

    if word1 in unigram_counter:
      word1_prob = unigram_counter[word1]/total

    if word2 in unigram_counter:
      word2_prob = unigram_counter[word2]/total

    if (word1,word2) in context_counter:
      pair_prob = context_counter[(word1,word2)]/total

    return math.log2(pair_prob/(word1_prob*word2_prob))

def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    word_vector = {}
    for word2 in frequent_unigrams:
      if (word1,word2) in context_counter:
        word_vector[word2] = pmi(word1, word2, unigram_counter, context_counter)
      else:
        word_vector[word2] = 0.0
    return word_vector

def get_top_k_dimensions(word1_vector, k):
    return dict(collections.Counter(word1_vector).most_common(k))

def get_cosine_similarity(word1_vector, word2_vector):
    A = []
    B = []
    for key, val in word1_vector.items():
      A.append(word1_vector[key])
      B.append(word2_vector[key])

    cosine = np.dot(A,B)/(norm(A)*norm(B))

    return cosine


def get_most_similar(word2vec, word, k):
  try:
    similar_words = word2vec.most_similar(word, topn=k)
    return similar_words
  except KeyError:
    # If the word is not in the vocabulary, return an empty list
    return []

def word_analogy(word2vec, word1, word2, word3):
    result = word2vec.most_similar(negative=[word1], positive=[word2, word3])
    return result[0]


def create_tfidf_matrix(documents, stopwords):

  #process docs
  tokenized_docs = []
  for doc in documents:
    doc = list(doc)
    tokens = []
    for word in doc:
      if word.lower() not in stopwords and word.isalnum():
        tokens.append(word.lower())
    tokenized_docs.append(tokens)

  vocab = []

  for lst in tokenized_docs:
    vocab += lst

  vocab = list(set(vocab))
  vocab.sort()

  #tf
  num_docs = len(list(documents))
  num_words = len(vocab)
  tf_matrix = np.zeros((num_docs, num_words))

  for i, doc in enumerate(tokenized_docs):
    for j, word in enumerate(vocab):
      wordcount = doc.count(word)
      tf_matrix[i][j] = wordcount

  #idf
  idf_vector = np.zeros(num_words)
  for j, word in enumerate(vocab):
    count = 0
    for doc in tokenized_docs:
      if word in doc:
        count+=1
    idf = math.log10(num_docs / (count + 1)) + 1
    idf_vector[j] = idf

  # Compute the TF-IDF matrix
  tfidf_matrix = np.zeros((num_docs, num_words))
  for i in range(num_docs):
    for j in range(num_words):
      tfidf = tf_matrix[i][j] * idf_vector[j]
      tfidf_matrix[i][j] = tfidf
    
  return tfidf_matrix, vocab


def get_idf_values(documents, stopwords):

    #process docs
    tokenized_docs = []
    for doc in documents:
      doc = list(doc)
      tokens = []
      for word in doc:
        if word.lower() not in stopwords and word.isalnum():
          tokens.append(word.lower())
      tokenized_docs.append(tokens)

    vocab = []
    for lst in tokenized_docs:
      vocab += lst

    vocab = list(set(vocab))
    vocab.sort()

    #tf
    num_docs = len(list(documents))
    num_words = len(vocab)

    #idf
    idf_dict = {}
    for j, word in enumerate(vocab):
      count = 0
      for doc in tokenized_docs:
        if word in doc:
          count+=1
      idf = math.log10(num_docs / count)
      idf_dict[word] = idf

    return idf_dict

def calculate_sparsity(tfidf_matrix):

    size = tfidf_matrix.size
    zero_counts = np.count_nonzero(tfidf_matrix == 0)
    return zero_counts / size

def extract_salient_words(VT, vocabulary, k):
  salient_words = {}
  for i in range(VT.shape[0]):
    row = VT[i,:]
    indexes = row.argsort()[::-1][:k]
    topk = [vocabulary[indx] for indx in indexes]
    salient_words[i] = topk
  return salient_words

def get_similar_documents(U, Sigma, VT, doc_index, k): 
  v = U[doc_index] * Sigma
  l = np.linalg.norm(v)
  similarities = []
  for i in range(U.shape[0]):
      v2 = U[i] * Sigma
      l2 = np.linalg.norm(v2)
      similarity = (v @ v2) / (l * l2)
      similarities.append(similarity)
  similarities = np.array(similarities)
  similar_indices = np.argsort(similarities)[::-1][1:k+1]
  return list(similar_indices)

def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):

  cleaned = []
  for word in query:
    if word.isalnum():
      cleaned.append(word.lower())

  tf_matrix = {}
  for j, word in enumerate(vocabulary):
      wordcount = cleaned.count(word)
      tf_matrix[word] = wordcount

  tfidf_matrix = np.zeros(len(vocabulary))
  for j, word in enumerate(vocabulary):
    tfidf = tf_matrix[word] * idf_values[word]
    tfidf_matrix[j] = tfidf

  query_ls = tfidf_matrix @ VT.T
  query_mini_norm = query_ls / np.linalg.norm(query_ls)
  
  l = np.linalg.norm(query_mini_norm)

  similarities = []
  for i in range(U.shape[0]):
    v2 = U[i] * Sigma
    l2 = np.linalg.norm(v2)
    similarity = query_mini_norm @ v2 / (l * l2)
    similarities.append(similarity)

  similarities = np.array(similarities)
  similar_indices = np.argsort(similarities)[::-1][:k]
  
  return list(similar_indices)

if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt') as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    
    ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    print(sample_output.most_common(10))
    """
    [(('the', 'pandemic'), 19811),
    (('a', 'pandemic'), 16615),
    (('a', 'mask'), 14353),
    (('a', 'wear'), 11017),
    (('wear', 'mask'), 10628),
    (('mask', 'wear'), 10628),
    (('do', 'n’t'), 10237),
    (('during', 'pandemic'), 8127),
    (('the', 'covid'), 7630),
    (('to', 'go'), 7527)]
    """
    ### END OF REFERENCE OUTPUT
    
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)

    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817


    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
