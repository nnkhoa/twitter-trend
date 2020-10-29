from os import listdir
from os.path import isfile, join
import sys 
import random

import jsonl_parser
import text_preprocess
import text_features
import data_seperation
import data_analyze

import pandas as pd


def load_raw_data(folder_path, trend_list, lang):
    # all_file = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    n_sample = 40

    list_file = [join(folder_path, f + ".jsonl") for f in trend_list if isfile(join(folder_path, f +  ".jsonl"))]

    data = []

    if n_sample > len(trend_list):
        n_sample = len(trend_list)

    for f in list_file[:n_sample]:
        data.extend(jsonl_parser.load_jsonl(f))

    print() 

    data_lang = [entry for entry in data if entry['lang'] == lang]

    return data_lang


def main():
    # data = load_raw_data(sys.argv[1])
    annotated = data_seperation.get_data('news')

    trend_list = annotated['id'].tolist()
    
    data = load_raw_data('/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/dataset/', trend_list, 'en')
    
    df = pd.DataFrame(data)

    df['trend_hash'].value_counts()

    df['text'] = df['text'].apply(lambda x: text_preprocess.remove_hyperlink(x))

    df_trend = df[['text', 'trend_hash']]

    trend_label = text_features.trend_mapping(set(df_trend['trend_hash'].tolist()))

    df_trend['label'] = df_trend.apply(lambda row: trend_label[row.trend_hash], axis=1)

    tweet_stats = data_analyze.tweet_length_stats(df)

    print(tweet_stats)

    ngram_freq = data_analyze.ngram_most_frequent(df)

    print(ngram_freq[:10])
    # df['text'] = df['text'].apply(lambda x: text_preprocess.tokenize_text(x))

    # print(df.text.to_string(index=False))

    # text_features.generate_co_occurrences_matrix(df['text'], 1)

    # tf, features = text_features.generate_term_freq(df['text'], 1)

    # text_features.generate_tfidf(df['text'])

    # text_features.lda_model(tf, features, 24)

    # text_features.k_mean_clustering(df_trend)
    
if __name__ == '__main__':
    main()