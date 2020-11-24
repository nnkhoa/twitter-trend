from sklearn.feature_extraction.text import CountVectorizer
import spacy
import en_core_web_sm 
import pandas as pd 
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import text_features

from collections import Counter
import itertools


def tweet_length_stats(master_data, trend_name=None):
    if trend_name is not None:
        temp_data = master_data.loc[master_data['trend_hash'] == trend_name]
        text_data = temp_data[['text']]
    else:
        text_data = master_data[['text']]

    if text_data.empty:
        max_length = 0
        min_length = 0
        avg_length = 0
    else:    
        text_data['length'] = text_data['text'].str.len()

        max_length = text_data['length'].max().item()

        min_length = text_data['length'].min().item()

        avg_length = text_data['length'].mean().item()

    return {"trend_name": trend_name, "max_length": max_length, "min_length": min_length, "avg_length": avg_length}


def ngram_most_frequent(master_data, n_gram=1, trend_name=None, plot=False):
    if trend_name is not None:
        temp_data = master_data.loc[master_data['trend_hash'] == trend_name]
        text_data = temp_data[['text']]
    else:
        text_data = master_data[['text']]

    tf_vector = CountVectorizer(ngram_range=(n_gram, n_gram),
                                max_df=0.95,
                                min_df=2,
                                lowercase=False,
                                # token_pattern=r'\S+',
                                stop_words='english')

    try:
        tf = tf_vector.fit_transform(text_data['text'])
    except ValueError:
        return {'trend_name': trend_name, 'ngram_freq': None}
    
    sum_ngram = tf.sum(axis=0)

    ngram_freq = {tuple(word.split(' ')) : sum_ngram[0, idx].item() for word, idx in tf_vector.vocabulary_.items()}

    ngram_freq = {k: v for k,v in sorted(ngram_freq.items(), key= lambda x: x[1], reverse=True)}

    top_10 = dict(Counter(ngram_freq).most_common(10))

    if plot == True:    
        key = top_10.keys()
        counts = list(top_10.values())
        key_string = list(map(lambda x: ' '.join(x), key))
        y_pos = np.arange(len(key_string))

        fig,ax = plt.subplots()
        
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(key_string)
        ax.invert_yaxis()
        ax.set_xlabel('# of mentions')
        ax.set_title(str(n_gram) + '-gram Frequencies')

        for i,v in enumerate(counts):
            ax.text(v + 3, i + .25, str(v), fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
    return {'trend_name': trend_name, 'ngram_freq': Counter(ngram_freq).most_common(10)}

    # features = tf_vector.get_feature_names()

    # text_features.print_top_word(tf, features, 10)


def most_named_entity(master_data, trend_name=None, plot=False):
    if trend_name is not None:
        temp_data = master_data.loc[master_data['trend_hash'] == trend_name]
        text_data = temp_data[['text']]
    else:
        text_data = master_data[['text']]

    ner = en_core_web_sm.load()

    text_data['ner'] = text_data['text'].apply(lambda x: list(ner(x).ents))

    ner_nested_list = text_data['ner'].tolist()

    ner_list = list(itertools.chain(*ner_nested_list))

    ner_list_text = list(itertools.chain([ner.text for ner in ner_list]))

    top = Counter(ner_list_text).most_common(20)

    print(top)

    if plot == True:    
        ne = [x[0] for x in top]
        counts = [x[1] for x in top]
        y_pos = np.arange(len(ne))

        fig,ax = plt.subplots()
        
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ne)
        ax.invert_yaxis()
        ax.set_xlabel('# of mentions')
        ax.set_title('Named Entity Frequencies')

        for i,v in enumerate(counts):
            ax.text(v + 3, i + .25, str(v), fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(top, columns=['NE', 'Count'])



