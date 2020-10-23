import pandas as pd

from os import listdir
from os.path import isfile, join, splitext

def get_data(type):
    annotation_path='/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/TT-annotations.csv'
    data_path='/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/dataset/'

    annotation_df = pd.read_csv(annotation_path, delimiter=';', names=['id', 'time', 'name', 'type'])
    news_df = annotation_df[annotation_df['type'] == type]

    list_file = [splitext(f)[0] for f in listdir(data_path) if isfile(join(data_path, f))]

    existing_trend = news_df[news_df['id'].isin(list_file)]

    return existing_trend