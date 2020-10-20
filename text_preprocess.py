import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sacremoses import MosesTokenizer

def text_preprocessing(text):
    # lowercasing
    text = text.lower()

    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # tokenization
    tokens = word_tokenize(text)

    # stopwords removal
    filtered_token = [word for word in tokens if word not in stopwords.words('english')]

    # stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(token) for token in filtered_token]

    return stemmed


def tokenize_text(text):
    mt = MosesTokenizer(lang='en')

    tokenized_text = mt.tokenize(text)

    return tokenized_text