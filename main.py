from os import listdir
from os.path import isfile, join
import sys 
import random
import json
import csv

import jsonl_parser
import text_preprocess
import text_features
import data_seperation
import data_analyze

import pandas as pd


def load_raw_data(folder_path, trend_list, lang, verbose):
    n_sample = 1036

    if trend_list:
        list_file = [join(folder_path, f + ".jsonl") for f in trend_list if isfile(join(folder_path, f +  ".jsonl"))]
        if n_sample > len(trend_list):
            n_sample = len(trend_list)
    else:
        list_file = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    
    data = []

    for f in list_file[:n_sample]:
        data.extend(jsonl_parser.load_jsonl(f, verbose))

    print() 

    data_lang = [entry for entry in data if entry['lang'] == lang]

    return data_lang


def main():
    verbose = False

    annotated = data_seperation.get_data('news')

    trend_list = annotated['id'].tolist()

    data_path='dataset/'
    
    data = load_raw_data(data_path, trend_list, 'en', verbose)
    
    df = pd.DataFrame(data)

    df['trend_hash'].value_counts()

    df['text'] = df['text'].apply(lambda x: text_preprocess.remove_hyperlink(x))

    df_trend = df[['text', 'trend_hash']]

    trend_label = text_features.trend_mapping(set(df_trend['trend_hash'].tolist()))

    df_trend['label'] = df_trend.apply(lambda row: trend_label[row.trend_hash], axis=1)

    # tweet_stats = []
    # for trend_name in trend_list:
    #     tweet_stats.append(data_analyze.tweet_length_stats(df, trend_name))

    # with open('tweet_stats.csv', 'w+') as fout:
    #     writer = csv.DictWriter(fout, fieldnames=['trend_name', 'max_length', 'min_length', 'avg_length'])
    #     writer.writeheader()
    #     for data in tweet_stats:
    #         writer.writerow(data)

    # print(data_analyze.tweet_length_stats(df))

    # ngram_freq = []
    # for trend_name in trend_list:
    #     ngram_freq.append(data_analyze.ngram_most_frequent(df, trend_name=trend_name))

    # with open('ngram_freq.json', 'w+') as fout:
    #     json.dump(ngram_freq, fout, indent=1)

    # print(data_analyze.ngram_most_frequent(df))

    # print(data_analyze.ngram_most_frequent(df, n_gram=2))

    data_analyze.most_named_entity(df)

    # ngram_freq = data_analyze.ngram_most_frequent(df)

    # df['text'] = df['text'].apply(lambda x: text_preprocess.tokenize_text(x))

    # print(df.text.to_string(index=False))

    # text_features.generate_co_occurrences_matrix(df['text'], 1)

    # tf, features = text_features.generate_term_freq(df['text'], 1)

    # text_features.generate_tfidf(df['text'])

    # text_features.lda_model(tf, features, 24)

    # text_features.k_mean_clustering(df_trend)
    
if __name__ == '__main__':
    main()