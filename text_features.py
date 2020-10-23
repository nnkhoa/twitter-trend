import itertools

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import metrics


def print_top_word(model, features_name, num_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " / ".join(features_name[i]
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
    tf_vector = CountVectorizer(ngram_range=(n_gram, n_gram),
                                max_df=0.95,
                                min_df=2, 
                                stop_words='english')

    tf = tf_vector.fit_transform(data)

    features_name = tf_vector.get_feature_names()

    return tf, features_name


def lda_model(tf, text_features, n_components):
    lda = LatentDirichletAllocation(n_components=n_components,
                                    max_iter=10, 
                                    learning_method='online', 
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)

    print_top_word(lda, text_features, 10)


def trend_mapping(trend_list):
    i = 0
    trend_dict = {}
    for trend in trend_list:
        trend_dict[trend] = i
        i+=1
    
    return trend_dict


def k_mean_clustering(data_frame):
    labels = data_frame['label'].tolist()
    true_k = len(set(labels))

    tfidf, terms = generate_tfidf(data_frame['text'])

    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=1)
    km.fit(tfidf)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
                        % metrics.adjusted_rand_score(labels, km.labels_))

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s /' % terms[ind], end='')
        print()

    # visualizing_data(km, tfidf)

def visualizing_data(model, data):
    #Mesh step size
    h = 0.02
    
    x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)