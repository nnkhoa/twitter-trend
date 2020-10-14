import jsonl_parser
import text_preprocess
import pandas as pd

def main():
    data = jsonl_parser.load_jsonl('G:\Gestion Action\GERANT\Khoa\data\Twitter\\107530150c943610b908a4c82168133d.jsonl')
    
    df = pd.DataFrame(data)

    df['text'] = df['text'].apply(lambda x: text_preprocess.text_preprocessing(x))

    print(df.text.to_string(index=False))

    
if __name__ == '__main__':
    main()