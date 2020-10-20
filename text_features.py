import itertools

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def generate_co_occurrences_matrix(data):
    count_vector = CountVectorizer()

    token_counts = count_vector.fit_transform(data)

    print(token_counts.shape)


