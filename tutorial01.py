import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Functions
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, "lxml") # removes xml tags
    letters_only = re.sub("^[a-zA-Z]", " ", review_text.get_text()) # leaves only letters (i.e. words)
    words = letters_only.lower().split() # to lower
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] # remove stopwords
    return(" ".join(meaningful_words)) # returns in a string



# Importing
train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

print train.shape
print train.columns.values

num_reviews = train["review"].size
clean_train_reviews = []
for i in xrange(0, num_reviews):
    if ((i+1)%1000 == 0):
        print "Review of %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append( review_to_words(train["review"][i]) )

# Vectorizer
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews) # returns 5000 x 25000 matrix

vocab = vectorizer.get_feature_names() # get list of vocab

train_data_features = train_data_features.toarray() # get the 5000 vector for each of the 25000 reviews
dist = np.sum(train_data_features, axis=0) # the number of times each word has appeared, summed up in 25000 reviews

# print number of counts for each word
#for tag, count in zip(vocab, dist):
#    print count, tag



# Random Forest-ing
print "Training random forest"
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train["sentiment"])


# Get test data
print "Reading test data, which has the shape: "
test = pd.read_csv("data/testData.tsv", header = 0, delimiter="\t", quoting = 3)
print test.shape

num_reviews = len(test["reviews"])
clean_test_reviews = []
for i in xrange(0, num_reviews):
    if ((i+1)%1000 == 0):
        print "Review of %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame( data = {"id": test["id"], "sentiment": result})
output.to_csv("Bag_of_Words_model_1.csv", index=False, quoting=3)
