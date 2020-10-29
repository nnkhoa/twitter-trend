import pandas as pd

from os import listdir
from os.path import isfile, join, splitext

def get_data(type):
    macbook_annotation_path='/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/TT-annotations.csv'
    macbook_data_path='/Users/khoanguyen/Workspace/dataset/twitter-trending/TT-classification/dataset/'

    workstation_annotation_path = 'C:\\Users\\nnguyen\\Documents\\Twitter\\TT-annotations.csv'
    workstation_data_path = 'C:\\Users\\nnguyen\\Documents\\Twitter\\dataset'

    annotation_df = pd.read_csv(workstation_annotation_path, delimiter=';', names=['id', 'time', 'name', 'type'])
    news_df = annotation_df[annotation_df['type'] == type]

    list_file = [splitext(f)[0] for f in listdir(workstation_data_path) if isfile(join(workstation_data_path, f))]

    existing_trend = news_df[news_df['id'].isin(list_file)]

    return existing_trend