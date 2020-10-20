from os import listdir
from os.path import isfile, join
import sys 
import random

import jsonl_parser
import text_preprocess
import text_features

import pandas as pd

n_sample = 100

def load_raw_data(folder_path):
    list_file = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    data = []

    for f in list_file[:100]:
        data.extend(jsonl_parser.load_jsonl(f))

    random.shuffle(data)

    return data


def main():
    # data = load_raw_data(sys.argv[1])
    data = load_raw_data('/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/dataset')
    
    df = pd.DataFrame(data)

    # df['text'] = df['text'].apply(lambda x: text_preprocess.tokenize_text(x))

    # print(df.text.to_string(index=False))

    text_features.generate_co_occurrences_matrix(df['text'])

    
if __name__ == '__main__':
    main()