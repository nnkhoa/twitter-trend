import pandas as pd

from os import listdir
from os.path import isfile, join, splitext

def get_data(trend_type):
    annotation_path='TT-annotations.csv'
    data_path='dataset-full/'

    annotation_df = pd.read_csv(annotation_path, delimiter=';', names=['id', 'time', 'name', 'type'])
    
    if trend_type is not None:
        trend_df = annotation_df[annotation_df['type'] == trend_type]
    else:
        trend_df = annotation_df
    
    list_file = [splitext(f)[0] for f in listdir(data_path) if isfile(join(data_path, f))]

    existing_trend = trend_df[trend_df['id'].isin(list_file)]

    return existing_trend