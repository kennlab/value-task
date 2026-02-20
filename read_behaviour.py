import pandas as pd
import json 

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

import numpy as np
def read_behaviour(file_path):
    data = read_jsonl(file_path)
    df = pd.DataFrame(data)
    df['accuracy'] = df['outcome'] == 'correct'
    df = df[~df['magnitudes'].isna()]
    # df['magnitudes'] = df['magnitudes'].apply(tuple)
    df['option1_mag'], df['option2_mag'] = np.array(df['magnitudes'].tolist()).T
    df['opt1left'] = df['condition'].str.contains('opt1left')
    df['left_mag'] = np.where(df['opt1left'], df['option1_mag'], df['option2_mag'])
    df['right_mag'] = np.where(~df['opt1left'], df['option1_mag'], df['option2_mag'])
    df['left_chosen'] = df['opt1left'] == df['accuracy']
    df['value_diff_lr'] = df['left_mag'] - df['right_mag']
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Read behavior data from a JSONL file.')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file containing behavior data.')
    args = parser.parse_args()

    df = read_behaviour(args.file_path)
    df = df.query('outcome!="timeout"')
    print('Trials completed:', len(df))
    print('Accuracy:', (df['outcome'] == 'correct').mean())
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = df.groupby('value_diff_lr').left_chosen.mean()
    ax.plot(x.index, x.values, marker='o')
    ax.set_xlabel('Value Difference (Left - Right)')
    ax.set_ylabel('Proportion')
    ax.set_title('Left Choice by Value Difference')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    plt.show()