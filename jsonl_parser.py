import json 
import os

def load_jsonl(input_file) -> list:
    data = []
    wanted_keys = ['created_at', 'id_str', 'text', 'lang', 'trend_hash']

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    
    print('Loaded {} records from {}'.format(len(data), input_file))

    trend_hash = os.path.splitext(os.path.basename(input_file))[0]
    
    for record in data:
        record.update({"trend_hash": trend_hash})

    # remove unwanted keys and return the list
    return [{k: record[k] for k in set(wanted_keys) & set(record.keys())} for record in data]

