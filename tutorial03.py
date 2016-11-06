from gensim.models import Word2Vec
import numpy as py

def MakeFeatureVec(words, model, num_features):
    print model
    print words
    featureVec = np.zeros((num_features), dtype="float32")
    nwords = 0
    index2word_set = set(model.index2word)
    print index2word_set
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


model = Word2Vec.load("300features_40minwords_10context")
