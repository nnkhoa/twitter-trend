import pickle
import pandas as pd
import datetime
from optparse import OptionParser 


def list_from_file(filename):
    if not filename:
        return None

    with open(filename) as fp:
	    content = fp.readlines()
	
    return [x.strip() for x in content]


def get_args():
    parser = OptionParser()

    parser.add_option("-t", "--tools", dest="tools",
                        help="Pick a tools to execute: request, sentiment")
    parser.add_option("--date-start", dest="date_start", default=None,
                        help="Starting Date of Query")
    parser.add_option("--date-end", dest="date_end", default=None,
                        help="End Date of Query")
    parser.add_option("--exclude-topics", dest="exclude_topics", default=None,
                        help="File containing list of topics to be excluded from querying")
    parser.add_option("--include-topics", dest="include_topics", default=None,
                        help="File containing list of topics to be included in querying")
    parser.add_option("--tickers", dest="tickers", default=None,
                        help="File containing list of tickers to be included in querying")
    parser.add_option("--keywords", dest="keywords", default=None,
                        help="File containing list of keywords to be included in querying")
    parser.add_option("--output", dest="output", default=None,
                        help="Output the data")
    (options, args) = parser.parse_args()

    return options


def date_parser(date_string):
    if not date_string:
        return None

    return datetime.datetime.strptime(date_string, '%Y-%m-%d')


def create_query_condition(options):
    return {'include_topics': list_from_file(options.include_topics), 
            'exclude_topics': list_from_file(options.exclude_topics), 
            'keywords': list_from_file(options.keywords), 
            'tickers': list_from_file(options.tickers), 
            'time': [date_parser(options.date_start), date_parser(options.date_end)]}


def save_as_pickle(object, filename, dump=1):
	if dump == 1:
		path = "G:\\Gestion Action\\GERANT\\Khoa\\data\\" + filename + ".pickle"
	else:
		path = filename
	
	with open(path, 'wb') as f:
		pickle.dump(object, f)


def read_pickle(filename, dump=1):
    if dump == 1:
        path = "G:\\Gestion Action\\GERANT\\Khoa\\data\\" + filename + ".pickle"
    else:
        path = filename
	
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df


def remove_duplicate(dataframe):
    is_duplicate = dataframe["headline"].duplicated(keep='last')

    return dataframe[~is_duplicate]


def percentage_digit(text):
    digits = sum(list(map(lambda x: 1 if x.isdigit() else 0, text)))  
    return digits/len(text)


def percentage_upper(text):
    uppers = sum(list(map(lambda x: 1 if x.isupper() else 0, text)))  
    return uppers/len(text)


def get_data(mongo_doc):
    topics = []
    for topic in mongo_doc.dtopics:
        topics.append({'Id': topic['Id'], 'score': topic['score']})

    tickers = []
    for ticker in mongo_doc.dtickers:
        tickers.append({'Id': ticker['Id'], 'score': ticker['score']})

    return {'time': mongo_doc.time, 'headline': mongo_doc.headline, 'text': mongo_doc.text, 'dtopics': topics, 'dtickers': tickers}


def generators_concat(generators_list):
    for generator in generators_list:
        yield from generator


def read_keyword(filename):
    with open(filename) as fp:
        content = fp.readlines()
	
    return [x.strip() for x in content]


def matching_ticker(doc_tickers, ticker_list):
    matching_tickers = []
    for ticker in doc_tickers:
        if int(ticker['score']) < 80:
            continue
        if ticker_list['EQY_FUND_TICKER'].str.contains(r'\b' + ticker['Id'] + r'\b').sum():
            matching_tickers.append(ticker['Id'])
    
    return matching_tickers


def fetch_ticker(input_data, ticker_list, output_name=None):
    text_df = input_data
    
    if isinstance(input_data, str):
        text_df = read_pickle(input_data)
    
    ticker_df = read_pickle(ticker_list)

    text_df['desired_ticker'] = text_df['dtickers'].apply(matching_ticker, args=[ticker_df])

    text_df.drop(text_df.index[text_df['desired_ticker'].str.len() == 0], inplace=True)

    if isinstance(output_name, str):
        save_as_pickle(text_df, output_name)

    return text_df