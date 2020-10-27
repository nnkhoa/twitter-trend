from sklearn.feature_extraction.text import CountVectorizer
import text_features

def tweet_length_stats(master_data, trend_name=None):
    if trend_name is not None:
        text_data = master_data.loc(master_data['id'] == trend_name)['text']
    else:
        text_data = master_data['text']

    text_data['length'] = text_data['text'].str.len()

    max_length = text_data['length'].max()

    min_length = text_data['length'].min()

    avg_length = text_data['length'].mean()

    return {"trend_name": trend_name, "max_length": max_length, "min_length": min_length, "avg_length": avg_length}


def ngram_most_frequent(master_data, n_gram=1, trend_name=None):
    if trend_name is not None:
        text_data = master_data.loc(master_data['id'] == trend_name)['text']
    else:
        text_data = master_data['text']

    tf_vector = CountVectorizer(ngram_range=(n_gram, n_gram),
                                max_df=0.95,
                                min_df=2, 
                                stop_words='english')

    tf = tf_vector.fit_transform(text_data['text'])

    features = tf.get_feature_names()

    text_features.print_top_word(tf, features, 10)


    




