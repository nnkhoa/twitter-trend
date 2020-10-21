import itertools

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def print_top_word(model, features_name, num_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join(features_name[i]
                            for i in topic.argsort()[:-num_words - 1:-1])
        print(message)
    
    print()


def generate_co_occurrences_matrix(data, n_gram):
    count_vector = CountVectorizer(ngram_range=(n_gram,n_gram), stop_words='english')

    X = count_vector.fit_transform(data)
    w_occur = (X.T * X)

    features_name = count_vector.get_feature_names()

    w_occur.setdiag(0)

    return w_occur, features_name


def generate_tfidf(data):
    tfidf_vector = TfidfVectorizer(stop_words='english')

    tfidf = tfidf_vector.fit_transform(data)

    features_name = tfidf_vector.get_feature_names()
    # print(tfidf)

    return tfidf, features_name

def generate_term_freq(data, n_gram):
    tf_vector = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words='english')

    tf = tf_vector.fit_transform(data)

    features_name = tf_vector.get_feature_names()

    return tf, features_name


def lda_model(tf, text_features):
    lda = LatentDirichletAllocation(max_iter=5, 
                                    learning_method='online', 
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)

    print_top_word(lda, text_features, 10)


    