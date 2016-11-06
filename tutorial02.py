import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec

# logging for word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Required because python couldn't read the pound symbol
# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# Functions
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    review = unicode(review.strip(), errors="replace")
    raw_sentences = tokenizer.tokenize(review.strip()) #split a paragraph into sentences
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist(raw_sentence, remove_stopwords)) #remove steop words and lower everything like before, then append it to sentences
    return sentences

def review_to_wordlist( review, remove_stopwords=False): # remove stop words and lower everything like before
    # Remove HTML tags
    review_text = BeautifulSoup(review, "lxml").get_text()
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # convert to lower and turn into list
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops] # remove stopwords
    return (words)


# MAIN

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )


sentences = []

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Sentence length: %d\n" % len(sentences)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Sentence length: %d\n" % len(sentences)


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print "Training model..."
model = word2vec.Word2Vec(sentences, workers = num_workers, size = num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace = True)
model_name = "300features_40minwords_10context"
model.save(model_name)

model.save_word2vec_format(model_name+".bin", binary=True)
model.save_word2vec_format(model_name+".txt", binary=False)
