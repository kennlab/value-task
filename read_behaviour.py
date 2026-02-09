import pandas as pd
import json 

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_behavior(file_path):
    data = read_jsonl(file_path)
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Read behavior data from a JSONL file.')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file containing behavior data.')
    args = parser.parse_args()

    df = read_behavior(args.file_path)
    df = df.query('outcome!="timeout"')
    print('Trials completed:', len(df))
    print('Accuracy:', (df['outcome'] == 'correct').mean())